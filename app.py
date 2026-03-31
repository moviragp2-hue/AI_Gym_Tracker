"""
AI Gym Tracker — Streamlit Real-Time App
Uses streamlit-webrtc for true browser-side webcam (works on Streamlit Cloud)
Supports: Squat · Push-Up · Pull-Up · Jumping Jack · Russian Twist
"""

import streamlit as st
import cv2
import numpy as np
import math
import os
import tempfile
import threading
from collections import deque
from typing import Tuple, Dict
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="AI Gym Tracker", page_icon="💪", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── Main app ── */
body, .stApp { background-color: #f5f6fa; color: #1a1a2e; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #dde1ea; }
[data-testid="stSidebar"] * { color: #1a1a2e !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown { color: #1a1a2e !important; }
[data-testid="stSidebar"] hr { border-color: #dde1ea !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label { color: #1a1a2e !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #f5f6fa !important; border-color: #dde1ea !important; color: #1a1a2e !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #1a1a2e !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { color: #1a1a2e !important; }
[data-testid="stSidebar"] .stButton > button { background-color: #f0f4f8 !important; color: #1a1a2e !important; border: 1px solid #dde1ea !important; border-radius: 10px !important; }
[data-testid="stSidebar"] .stButton > button:hover { background-color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: #16a34a !important; color: #ffffff !important; border: none !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background-color: #15803d !important; }

/* ── Metric cards ── */
.metric-card { background:#ffffff; border-radius:14px; padding:18px 22px; border:1px solid #dde1ea; box-shadow:0 2px 8px rgba(0,0,0,0.06); text-align:center; margin-bottom:10px; }
.metric-value { font-size:3rem; font-weight:800; color:#16a34a; line-height:1.1; }
.metric-label { font-size:0.82rem; color:#6b7280; letter-spacing:1.5px; text-transform:uppercase; margin-top:4px; }
.feedback-box { background:#ffffff; border-left:4px solid #16a34a; border-radius:10px; padding:12px 18px; margin-top:10px; font-size:1.05rem; color:#1a1a2e; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.score-bar-bg { background:#e5e7eb; border-radius:8px; height:14px; margin:6px 0; }
.score-bar-fill { background:#16a34a; height:14px; border-radius:8px; }
h1 { color:#16a34a !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "pose_landmarker_heavy.task"

@st.cache_resource(show_spinner="Downloading MediaPipe model (~25 MB)…")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    return MODEL_PATH

ensure_model()

LM = {'l_shoulder':11,'r_shoulder':12,'l_elbow':13,'r_elbow':14,'l_wrist':15,'r_wrist':16,
      'l_hip':23,'r_hip':24,'l_knee':25,'r_knee':26,'l_ankle':27,'r_ankle':28}
SKELETON_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def lm_px(lms, idx, w, h): lm=lms[idx]; return int(lm.x*w), int(lm.y*h)
def angle3(a,b,c):
    a,b,c=np.array(a,float),np.array(b,float),np.array(c,float)
    ba,bc=a-b,c-b
    return math.degrees(math.acos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8),-1,1)))
def mid(p1,p2): return (int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2))

def _angle_score(best, ideal, worst):
    """Map best angle of a rep to 0-100. ideal=100, worst=0."""
    if ideal < worst:  # lower is better (squat, push-up, pull-up)
        return max(0, min(100, int((worst - best) / (worst - ideal) * 100)))
    else:              # higher is better (jumping jack)
        return max(0, min(100, int((best - worst) / (ideal - worst) * 100)))

class RepCounter:
    def __init__(self,up_thresh,down_thresh,higher_is_up=True):
        self.up_thresh=up_thresh; self.down_thresh=down_thresh; self.higher_is_up=higher_is_up
        self.count=0; self.stage=None; self.history=deque(maxlen=2000)
        self._rep_best=None; self._angle_scores=[]
    def update(self,v):
        self.history.append(v)
        if self._rep_best is None: self._rep_best=v
        if self.higher_is_up:
            self._rep_best=min(self._rep_best,v)
            if v>self.up_thresh: self.stage='up'; self._rep_best=None
            elif v<self.down_thresh and self.stage=='up':
                self.stage='down'; self.count+=1; return 'rep'
        else:
            self._rep_best=max(self._rep_best,v)
            if v<self.up_thresh: self.stage='up'; self._rep_best=None
            elif v>self.down_thresh and self.stage=='up':
                self.stage='down'; self.count+=1; return 'rep'
        return ''
    def angle_score(self): return int(sum(self._angle_scores)/len(self._angle_scores)) if self._angle_scores else 0
    def reset(self): self.count=0; self.stage=None; self.history.clear(); self._rep_best=None; self._angle_scores=[]

class SquatAnalyzer:
    label='Squat'
    def __init__(self): self.rc=RepCounter(160,90)
    def analyze(self,lms,w,h):
        hip=lm_px(lms,LM['l_hip'],w,h); knee=lm_px(lms,LM['l_knee'],w,h); ankle=lm_px(lms,LM['l_ankle'],w,h)
        ang=angle3(hip,knee,ankle); rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, ideal=70, worst=160))
        fb='Go lower!' if ang>110 and self.rc.stage!='down' else ('Good depth! 🔥' if ang<90 else 'Stand tall!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PushUpAnalyzer:
    label='Push-Up'
    def __init__(self): self.rc=RepCounter(155,90)
    def analyze(self,lms,w,h):
        sh=lm_px(lms,LM['l_shoulder'],w,h); el=lm_px(lms,LM['l_elbow'],w,h); wr=lm_px(lms,LM['l_wrist'],w,h)
        ang=angle3(sh,el,wr); rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, ideal=60, worst=155))
        fb='Lower chest!' if ang>130 and self.rc.stage!='down' else ('Good depth! 🔥' if ang<90 else 'Push up!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class PullUpAnalyzer:
    label='Pull-Up'
    def __init__(self): self.rc=RepCounter(50,140,higher_is_up=False)
    def analyze(self,lms,w,h):
        sh=lm_px(lms,LM['l_shoulder'],w,h); el=lm_px(lms,LM['l_elbow'],w,h); wr=lm_px(lms,LM['l_wrist'],w,h)
        ang=angle3(sh,el,wr); rep=self.rc.update(ang)
        if rep=='rep' and self.rc._rep_best is not None:
            self.rc._angle_scores.append(_angle_score(self.rc._rep_best, ideal=30, worst=140))
        fb='Pull higher!' if ang>70 and self.rc.stage=='up' else ('Chin over bar! 🔥' if ang<50 else 'Lower slowly!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class JumpingJackAnalyzer:
    label='Jumping Jack'
    def __init__(self): self.rc=RepCounter(130,40); self._rep_max=0
    def analyze(self,lms,w,h):
        lsh=lm_px(lms,LM['l_shoulder'],w,h); lhi=lm_px(lms,LM['l_hip'],w,h); lwr=lm_px(lms,LM['l_wrist'],w,h)
        ang=angle3(lhi,lsh,lwr)
        self._rep_max=max(self._rep_max,ang)
        rep=self.rc.update(ang)
        if rep=='rep':
            self.rc._angle_scores.append(_angle_score(self._rep_max, ideal=150, worst=40))
            self._rep_max=0
        fb='Arms up!' if ang<80 else ('Great! 🔥' if ang>120 else 'Keep going!')
        return dict(angle=ang,stage=self.rc.stage,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

class RussianTwistAnalyzer:
    label='Russian Twist'
    def __init__(self): self.rc=RepCounter(30,5); self._last_side=None; self._touches=0; self._touch_max=0
    def analyze(self,lms,w,h):
        lsh=lm_px(lms,LM['l_shoulder'],w,h); rsh=lm_px(lms,LM['r_shoulder'],w,h)
        lhi=lm_px(lms,LM['l_hip'],w,h); rhi=lm_px(lms,LM['r_hip'],w,h)
        lwr=lm_px(lms,LM['l_wrist'],w,h); rwr=lm_px(lms,LM['r_wrist'],w,h)
        scx=(lsh[0]+rsh[0])/2; hcx=(lhi[0]+rhi[0])/2; rot=abs(scx-hcx)
        wcx=(lwr[0]+rwr[0])/2; side='left' if wcx<hcx else 'right'
        self._touch_max=max(self._touch_max,rot)
        if rot>30 and side!=self._last_side:
            self._last_side=side; self._touches+=1
            if self._touches%2==0:
                self.rc.count+=1
                self.rc._angle_scores.append(_angle_score(self._touch_max, ideal=80, worst=0))
                self._touch_max=0
        self.rc.history.append(rot)
        fb=f'Twist {side}!' if rot<20 else f'Good twist → {side}! 🔥'
        return dict(angle=rot,stage=side,count=self.rc.count,feedback=fb,form_score=self.rc.angle_score())

ANALYZERS = {'Squat':SquatAnalyzer,'Push-Up':PushUpAnalyzer,'Pull-Up':PullUpAnalyzer,
             'Jumping Jack':JumpingJackAnalyzer,'Russian Twist':RussianTwistAnalyzer}

FONT = cv2.FONT_HERSHEY_DUPLEX

def draw_skeleton(frame, lms, w, h):
    for a,b in SKELETON_EDGES:
        cv2.line(frame,lm_px(lms,a,w,h),lm_px(lms,b,w,h),(34,139,34),2,cv2.LINE_AA)
    for i in range(33):
        pt=lm_px(lms,i,w,h); cv2.circle(frame,pt,5,(74,163,22),-1,cv2.LINE_AA); cv2.circle(frame,pt,5,(255,255,255),1,cv2.LINE_AA)

def draw_hud(frame, res, ex_title):
    H,W=frame.shape[:2]; count=res.get('count',0); stage=res.get('stage','') or ''; feedback=res.get('feedback',''); angle_v=float(res.get('angle') or 0)
    label=ex_title.upper(); lw,_=cv2.getTextSize(label,FONT,0.65,1)[0]
    cv2.rectangle(frame,(W//2-lw//2-12,6),(W//2+lw//2+12,38),(230,232,235),-1)
    cv2.putText(frame,label,(W//2-lw//2,30),FONT,0.65,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,label,(W//2-lw//2,30),FONT,0.65,(74,163,22),1,cv2.LINE_AA)
    cv2.rectangle(frame,(8,48),(138,178),(230,232,235),-1); cv2.rectangle(frame,(8,48),(138,52),(74,163,22),-1)
    cv2.putText(frame,'REPS',(18,73),FONT,0.42,(100,108,120),1,cv2.LINE_AA)
    cstr=str(count); cw,_=cv2.getTextSize(cstr,FONT,2.6,2)[0]
    cv2.putText(frame,cstr,(73-cw//2,158),FONT,2.6,(0,0,0),5,cv2.LINE_AA); cv2.putText(frame,cstr,(73-cw//2,158),FONT,2.6,(74,163,22),2,cv2.LINE_AA)
    sl=stage.upper() if stage else 'READY'; sc=(74,163,22) if stage=='up' else (235,99,37)
    cv2.rectangle(frame,(8,186),(138,212),sc,-1); sw,_=cv2.getTextSize(sl,FONT,0.42,1)[0]
    cv2.putText(frame,sl,(73-sw//2,205),FONT,0.42,(0,0,0),2,cv2.LINE_AA)
    cv2.rectangle(frame,(W-148,48),(W-8,178),(230,232,235),-1); cv2.rectangle(frame,(W-148,48),(W-8,52),(235,99,37),-1)
    cv2.putText(frame,'ANGLE',(W-138,73),FONT,0.42,(100,108,120),1,cv2.LINE_AA)
    astr=f'{int(angle_v)} deg'; aw,_=cv2.getTextSize(astr,FONT,0.75,1)[0]
    cv2.putText(frame,astr,(W-78-aw//2,130),FONT,0.75,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,astr,(W-78-aw//2,130),FONT,0.75,(235,99,37),1,cv2.LINE_AA)
    if feedback:
        fb=feedback[:58]; fw,fh=cv2.getTextSize(fb,FONT,0.5,1)[0]; fy2=H-12; fy1=fy2-fh-16
        cv2.rectangle(frame,(W//2-fw//2-18,fy1),(W//2+fw//2+18,fy2),(230,232,235),-1)
        cv2.putText(frame,fb,(W//2-fw//2,fy2-7),FONT,0.5,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,fb,(W//2-fw//2,fy2-7),FONT,0.5,(255,255,255),1,cv2.LINE_AA)

class GymState:
    def __init__(self):
        self.lock=threading.Lock(); self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
        self.exercise='Squat'; self.show_skeleton=True; self.mirror=True
        self.analyzer=SquatAnalyzer(); self._mp=None; self._landmarker=None
    def set_exercise(self,ex):
        with self.lock:
            if ex!=self.exercise:
                self.exercise=ex; self.analyzer=ANALYZERS[ex]()
                self.result={'count':0,'stage':'','feedback':'Get in position!','angle':0,'form_score':0}
                self._close_landmarker()
    def _close_landmarker(self):
        if self._landmarker:
            try: self._landmarker.close()
            except: pass
            self._landmarker=None
    def get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            self._mp=mp
            opts=PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionTaskRunningMode.IMAGE,num_poses=1,
                min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5)
            self._landmarker=PoseLandmarker.create_from_options(opts)
        return self._mp, self._landmarker
    def process_frame(self, frame_bgr):
        with self.lock:
            mp,landmarker=self.get_landmarker()
            H,W=frame_bgr.shape[:2]
            rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            det=landmarker.detect(mp_img)
            if det.pose_landmarks:
                lms=det.pose_landmarks[0]
                if self.show_skeleton: draw_skeleton(frame_bgr,lms,W,H)
                self.result=self.analyzer.analyze(lms,W,H)
            else:
                self.result={**self.result,'feedback':'No pose — step back & stand tall'}
            draw_hud(frame_bgr,self.result,self.exercise)
            return frame_bgr

if 'gym_state' not in st.session_state:
    st.session_state.gym_state = GymState()
gym = st.session_state.gym_state
if 'redirect_score' not in st.session_state:
    st.session_state.redirect_score = None

with st.sidebar:
    st.markdown("## 💪 AI Gym Tracker")
    st.markdown("---")
    exercise = st.selectbox("Exercise", list(ANALYZERS.keys()))
    st.markdown("---")
    mode = st.radio("Mode", ["📹 Webcam (Real-Time)", "📁 Upload Video"], label_visibility="collapsed")
    st.markdown("---")
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    mirror = st.checkbox("Mirror Webcam", value=True)
    st.markdown("---")
    if st.button("🔄 Reset Counter", use_container_width=True):
        with gym.lock:
            gym.analyzer.rc.reset()
            gym.result={'count':0,'stage':'','feedback':'Ready!','angle':0,'form_score':0}
        st.rerun()
    st.markdown("---")
    st.markdown("**Exercises:**\n🦵 Squat · 💪 Push-Up\n🏋️ Pull-Up · 🙆 Jumping Jack\n🔄 Russian Twist")

gym.set_exercise(exercise)
gym.show_skeleton = show_skeleton
gym.mirror = mirror

st.title("💪 AI Gym Tracker — Real-Time")
st.caption("MediaPipe Pose Estimation · Select exercise · Allow camera when prompted")

# ── Redirect handler ──
col_vid, col_stats = st.columns([3,1])

with col_stats:
    rep_ph=st.empty(); stage_ph=st.empty(); angle_ph=st.empty(); fb_ph=st.empty()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    end_btn_ph=st.empty()

def render_stats():
    res=gym.result; cnt=res.get('count',0); s=(res.get('stage','') or 'READY').upper()
    ang=int(res.get('angle') or 0); fb=res.get('feedback',''); sc100=res.get('form_score',0)
    sc='#16a34a' if s in ('UP','READY') else '#2563eb'
    rep_ph.markdown(f'<div class="metric-card"><div class="metric-value">{cnt}</div><div class="metric-label">Reps</div></div>',unsafe_allow_html=True)
    stage_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem;color:{sc};">{s}</div><div class="metric-label">Stage</div></div>',unsafe_allow_html=True)
    angle_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#28b9ff;">{ang}°</div><div class="metric-label">Angle</div></div>',unsafe_allow_html=True)
    if fb: fb_ph.markdown(f'<div class="feedback-box">💬 {fb}</div>',unsafe_allow_html=True)
    _url = f"http://localhost/movera/patient/patient-plan.php?score={sc100}"
    end_btn_ph.markdown(
        f'''<a href="{_url}" target="_blank" rel="noopener noreferrer"
            style="display:block;width:100%;padding:10px 0;background:#16a34a;color:#fff;
                   text-align:center;border-radius:10px;font-weight:700;font-size:1rem;
                   text-decoration:none;margin-top:8px;">
            End Exercise (Score: {sc100})
        </a>''',
        unsafe_allow_html=True
    )

render_stats()

if mode.startswith("📹"):
    with col_vid:
        st.markdown("#### 📹 Live Webcam — Real-Time Tracking")
        st.info("Click **START** → allow camera access → begin exercising!")

    def video_frame_callback(frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if gym.mirror: img = cv2.flip(img, 1)
        try: img = gym.process_frame(img)
        except Exception as e: cv2.putText(img,f"Err:{str(e)[:35]}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        return VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIG = RTCConfiguration({"iceServers":[
        {"urls":["stun:stun.l.google.com:19302"]},
        {"urls":["stun:stun1.l.google.com:19302"]},
    ]})

    with col_vid:
        ctx = webrtc_streamer(
            key=f"gym-{exercise}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video":{"width":640,"height":480},"audio":False},
            async_processing=True,
        )

    if ctx.state.playing:
        import time
        while ctx.state.playing:
            render_stats()
            time.sleep(0.25)

else:
    with col_vid:
        st.markdown("#### 📁 Upload Video for Analysis")
        uploaded = st.file_uploader("MP4 / AVI / MOV", type=["mp4","avi","mov","mkv"])
        if uploaded:
            st.video(uploaded)
            if st.button("🧠 Analyze Video", type="primary", use_container_width=True):
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
                with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as f:
                    f.write(uploaded.read()); tmp_path=f.name
                opts=PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionTaskRunningMode.VIDEO,num_poses=1,
                    min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,min_tracking_confidence=0.5)
                lm=PoseLandmarker.create_from_options(opts)
                cap=cv2.VideoCapture(tmp_path)
                vw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                analyzer=ANALYZERS[exercise](); prog=st.progress(0,"Processing…"); preview=st.empty()
                last_res={'count':0,'stage':'','feedback':'','angle':0}; fidx=0
                while True:
                    ret,frame=cap.read()
                    if not ret: break
                    fidx+=1
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    det=lm.detect_for_video(mp_img,int(fidx/fps*1000))
                    if det.pose_landmarks:
                        lms=det.pose_landmarks[0]
                        if show_skeleton: draw_skeleton(frame,lms,vw,vh)
                        last_res=analyzer.analyze(lms,vw,vh)
                    else: last_res={**last_res,'feedback':'No pose detected'}
                    draw_hud(frame,last_res,exercise)
                    if fidx%30==0 or fidx==tot:
                        preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
                        prog.progress(min(fidx/max(tot,1),1.0),text=f"Frame {fidx}/{tot} — Reps: {last_res['count']}")
                cap.release(); lm.close(); os.unlink(tmp_path); prog.empty()
                st.success(f"✅ Done! **{last_res['count']} reps** in {fidx} frames.")
                gym.result=last_res; render_stats()
                hist=list(analyzer.rc.history)
                if hist:
                    import pandas as pd
                    st.markdown("#### 📈 Angle History")
                    st.line_chart(pd.DataFrame({'angle':hist}),color="#64ffa0")
