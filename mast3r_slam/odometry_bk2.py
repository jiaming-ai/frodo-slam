#!/usr/bin/env python3
"""
Straight‑or‑spin odometry *with heading fusion* (wheel ⊕ vision).

New  ▸ EKF on θ (yaw)
     ▸ wheel reading = prediction step
     ▸ vision angle  = update step
"""

from __future__ import annotations
import base64, math, threading, time
from typing import List, Tuple

import cv2, numpy as np, requests


class StraightOrSpinOdometry:
    # ‑‑‑ geometry / params ‑‑‑------------------------------------------------
    _DIAM_M   = 0.095
    _TRACK_M  = 0.160
    _CIRC_M   = math.pi * _DIAM_M
    _RPM_EQ_EPS = 0.5
    _FEATURES_MAX = 200

    # KF noise (tune on your robot)
    _SIGMA_OMEGA = 0.08       # rad s⁻¹ process noise from wheel drift
    _SIGMA_VIZ   = 0.02       # rad   vision measurement noise

    # ------------------------------------------------------------------------
    def __init__(self,
                 rpm_api = "http://localhost:8000/data",
                 cam_api = "http://localhost:8000/v2/front",
                 poll_s  = 0.1, timeout_s = 2.0):
        self._rpm_api, self._cam_api = rpm_api, cam_api
        self._poll_s,  self._timeout = poll_s,  timeout_s

        # pose & trajectory
        self._x = self._y = 0.0
        self._th = 0.0        # ----- EKF state
        self._P  = 0.05**2    # covariance of θ
        self._path: List[Tuple[float,float]] = [(0.0,0.0)]
        self._lock = threading.Lock()

        # integration memory
        self._prev_frame: np.ndarray | None = None
        self._prev_ts: float | None = None

        self._running=False
        self._thread: threading.Thread|None=None

    # ------------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, join: bool = True):
        self._running = False
        if join and self._thread:
            self._thread.join()

    def get_pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._x, self._y, self._th
    # ------------ helpers ---------------------------------------------------
    @staticmethod
    def _wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
    @staticmethod
    def _rpm_to_mps(rpm): return rpm/60.0*StraightOrSpinOdometry._CIRC_M

    # ——— vision yaw (unchanged except returns Δθ) ——————————————
    def _yaw_from_frames(self, prev, cur)->float|None:
        g0=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
        g1=cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        p0=cv2.goodFeaturesToTrack(g0,self._FEATURES_MAX,0.01,8)
        if p0 is None or len(p0)<6: return None
        p0=p0.reshape(-1,2)
        p1,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,p0,None)
        if p1 is None or st is None: return None
        p1=p1.reshape(-1,2); mask=st.ravel()==1
        if mask.sum()<6: return None
        g0p,g1p=p0[mask],p1[mask]
        h,w=g0.shape; c=np.array([w*0.5,h*0.5],np.float32)
        v0,v1=g0p-c, g1p-c
        cross=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
        dot=(v0*v1).sum(1)
        ang=np.arctan2(cross,dot)
        return float(np.median(ang))

    # ——— EKF update helper ————————————————————————————————
    def _kf_predict(self, omega, dt):
        # θ ← θ + ω·dt
        self._th = self._wrap(self._th + omega*dt)
        self._P += (self._SIGMA_OMEGA*dt)**2          # P = P+Q

    def _kf_update(self, theta_meas):
        # z = θ + v,  v~N(0,R)
        R = self._SIGMA_VIZ**2
        K = self._P / (self._P + R)                   # Kalman gain
        self._th = self._wrap(self._th + K*self._wrap(theta_meas - self._th))
        self._P  = (1-K)*self._P

    # ——— main loop ——————————————————————————————————————————
    def _loop(self):
        while self._running:
            try:
                rpm_rows = requests.get(self._rpm_api,timeout=self._timeout)\
                                      .json().get("rpms",[])
                rpm_rows.sort(key=lambda r:r[4])

                cam_resp = requests.get(self._cam_api,timeout=self._timeout).json()
                frame=None
                if cam_resp.get("front_frame"):
                    frame=cv2.imdecode(
                        np.frombuffer(base64.b64decode(cam_resp["front_frame"]),np.uint8),
                        cv2.IMREAD_COLOR)

                for r1,r2,r3,r4,ts in rpm_rows:
                    if self._prev_ts is not None and ts<=self._prev_ts: continue
                    dt = 0.0 if self._prev_ts is None else ts-self._prev_ts
                    self._prev_ts = ts

                    rpm_l,rpm_r = 0.5*(r1+r3), 0.5*(r2+r4)
                    v_l,v_r = self._rpm_to_mps(rpm_l), self._rpm_to_mps(rpm_r)
                    omega = (v_r - v_l)/self._TRACK_M     # rad/s

                    # ---------- EKF prediction ----------
                    self._kf_predict(omega, dt)

                    # ---------- integrate translation ----------
                    v = 0.5*(v_l+v_r)
                    with self._lock:
                        self._x += v*math.cos(self._th)*dt
                        self._y += v*math.sin(self._th)*dt
                        self._path.append((self._x,self._y))

                # vision update **once per loop** (frame pair ready)
                if frame is not None and self._prev_frame is not None:
                    dth = self._yaw_from_frames(self._prev_frame, frame)
                    if dth is not None:
                        self._kf_update(self._wrap(self._th + dth))
                if frame is not None:
                    self._prev_frame = frame

            except Exception as e:
                print("[Odom] warn:",e)
            time.sleep(self._poll_s)

    # ----------------------  realtime visualiser  --------------------------
    def visualize(self, win="Odo", scale_px=100, size=800):
        """
        Pop up an OpenCV window that shows the path and current heading.
        Press **q** to quit.
        
        Robot control:
        - **w** - move forward
        - **s** - move backward
        - **a** - rotate left
        - **d** - rotate right
        - **+/=** - increase speed
        - **-** - decrease speed
        """
        import cv2, numpy as np, math
        cv2.namedWindow(win)
        org=(size//2,size//2)
        
        # Robot control settings
        robot_speed = 0.5
        last_key_time = 0
        key_repeat_interval = 0.1  # seconds between repeated commands
        key_states = {ord("w"): False, ord("a"): False, ord("s"): False, ord("d"): False}
        
        while True:
            img=np.zeros((size,size,3),np.uint8)
            cv2.line(img,(0,org[1]),(size,org[1]),(40,40,40),1)
            cv2.line(img,(org[0],0),(org[0],size),(40,40,40),1)
            with self._lock:
                pts=self._path[-2000:]; x,y,th=self._x,self._y,self._th
            for (ax,ay),(bx,by) in zip(pts[:-1],pts[1:]):
                p1=(int(org[0]+ax*scale_px),int(org[1]-ay*scale_px))
                p2=(int(org[0]+bx*scale_px),int(org[1]-by*scale_px))
                cv2.line(img,p1,p2,(0,255,0),2)
            tip=(int(org[0]+x*scale_px),int(org[1]-y*scale_px))
            head=(int(tip[0]+30*math.cos(th)),int(tip[1]-30*math.sin(th)))
            cv2.arrowedLine(img,tip,head,(0,0,255),2,tipLength=0.3)
            
            # Add control instructions to the visualization
            instructions = [
                "Robot Control:",
                "w - forward",
                "s - backward", 
                "a - rotate left",
                "d - rotate right",
                "+/= - increase speed",
                "- - decrease speed",
                f"Speed: {robot_speed:.1f}"
            ]
            
            for i, text in enumerate(instructions):
                cv2.putText(img, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow(win,img)
            key = cv2.waitKey(30) & 0xFF
            
            # Handle key press events
            current_time = time.time()
            if key == ord("q"):
                break
            
            # Check for key presses and releases
            if key != 255:  # A key is pressed
                if key in key_states:
                    # Set this key as pressed, reset others
                    for k in key_states:
                        key_states[k] = (k == key)
                elif key == ord("+") or key == ord("="):
                    robot_speed = min(1.0, robot_speed + 0.1)
                elif key == ord("-"):
                    robot_speed = max(0.1, robot_speed - 0.1)
            else:  # No key is pressed (255)
                # Reset all key states when no key is pressed
                for k in key_states:
                    key_states[k] = False
                self._send_robot_command("stop", 0)
            
            # Process currently pressed keys
            if current_time - last_key_time >= key_repeat_interval:
                if any(key_states.values()):
                    last_key_time = current_time
                    if key_states[ord("w")]:
                        self._send_robot_command("forward", robot_speed)
                    elif key_states[ord("s")]:
                        self._send_robot_command("backward", robot_speed)
                    elif key_states[ord("a")]:
                        self._send_robot_command("left", robot_speed)
                    elif key_states[ord("d")]:
                        self._send_robot_command("right", robot_speed)

        # Make sure to stop the robot when closing the window
        self._send_robot_command("stop", 0)
        cv2.destroyWindow(win)
    
    def _send_robot_command(self, command, speed=0.5):
        """
        Send a command to the robot using the control API.
        
        Args:
            command: Direction to move ("forward", "backward", "left", "right", "stop")
            speed: Speed factor between 0 and 1
        """
        linear, angular = 0, 0
        
        if command == "forward":
            linear = speed
        elif command == "backward":
            linear = -speed
        elif command == "left":
            angular = speed
        elif command == "right":
            angular = -speed
        # "stop" command keeps linear and angular at 0
        else:
            return
        
        try:
            response = requests.post(
                'http://localhost:8000/control',
                json={'command': {'linear': linear, 'angular': angular}}
            )
            if response.status_code == 200:
                print(f"Robot command sent: {command} (linear={linear}, angular={angular})")
            else:
                print(f"Failed to send robot command: {response.status_code}")
        except Exception as e:
            print(f"Error sending robot command: {e}")

# --------------------------------------------------------------------------
if __name__=="__main__":
    odo=StraightOrSpinOdometry()
    odo.start()
    try: odo.visualize()
    finally: odo.stop()