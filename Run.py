#!/usr/bin/env python3

import subprocess, sys, time, math, random, collections

for pkg in ["opencv-python", "mediapipe", "numpy"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import cv2
import mediapipe as mp
import numpy as np

# ── Layout ───────────────────────────────────────────────────────────────────
W      = 540
VIZ_H  = 480
CAM_H  = 400
BAR_H  = 30
TOTAL_H = VIZ_H + CAM_H + BAR_H

mp_hands    = mp.solutions.hands
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# ── Vectorized projection (numpy arrays in, numpy arrays out) ─────────────────
# Projects Nx3 points → Nx2 pixel coords
CX, CY   = 270, 240     # center of viz panel
SCALE    = 195           # bigger = larger cube/tree
EYE_Z    = 3.5
RX_FIXED = -0.18         # fixed slight downward tilt

def proj_batch(pts_3d, ry):
    """pts_3d: (N,3) float array. Returns (N,2) int array."""
    x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
    # Y-axis rotation
    x2 =  x*math.cos(ry) + z*math.sin(ry)
    z2 = -x*math.sin(ry) + z*math.cos(ry)
    # X-axis tilt
    rx = RX_FIXED
    y3 =  y*math.cos(rx) - z2*math.sin(rx)
    z3 =  y*math.sin(rx) + z2*math.cos(rx)
    dz = np.maximum(EYE_Z - z3, 0.01)
    px = (CX + x2/dz*SCALE).astype(np.int32)
    py = (CY - y3/dz*SCALE).astype(np.int32)
    return np.stack([px, py], axis=1)

def proj_single(x, y, z, ry):
    x2 =  x*math.cos(ry) + z*math.sin(ry)
    z2 = -x*math.sin(ry) + z*math.cos(ry)
    rx = RX_FIXED
    y3 =  y*math.cos(rx) - z2*math.sin(rx)
    z3 =  y*math.sin(rx) + z2*math.cos(rx)
    dz = max(EYE_Z - z3, 0.01)
    return int(CX + x2/dz*SCALE), int(CY - y3/dz*SCALE)

# ── Cube ─────────────────────────────────────────────────────────────────────
CUBE_V = np.array([[-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
                   [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1]], dtype=np.float32)*0.95
CUBE_E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def draw_cube(canvas, ry):
    pts = proj_batch(CUBE_V, ry)
    for a, b in CUBE_E:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (200,200,200), 1, cv2.LINE_AA)
    for p in pts:
        cv2.rectangle(canvas,(p[0]-5,p[1]-5),(p[0]+5,p[1]+5),(255,255,255),-1)

def draw_grid_floor(canvas, ry, n=8):
    sc = 0.95; step = 2*sc/n; y = -sc
    for i in range(n+1):
        xi = -sc + i*step
        p0 = proj_single(xi, y, -sc, ry)
        p1 = proj_single(xi, y,  sc, ry)
        cv2.line(canvas, p0, p1, (40,40,40), 1, cv2.LINE_AA)
        p0 = proj_single(-sc, y, xi, ry)
        p1 = proj_single( sc, y, xi, ry)
        cv2.line(canvas, p0, p1, (40,40,40), 1, cv2.LINE_AA)

# ── Tree generation (runs once, stored as numpy array) ────────────────────────
def _grow(pos, dirn, length, depth, maxd, pts, rng):
    if depth > maxd or length < 0.008: return
    steps  = max(6, int(length * 80))
    spread = 0.014*(1 - depth/max(maxd,1))
    for s in range(steps+1):
        t = s/max(steps,1)
        pts.append([pos[0]+dirn[0]*length*t + rng.gauss(0,spread),
                    pos[1]+dirn[1]*length*t + rng.gauss(0,spread),
                    pos[2]+dirn[2]*length*t + rng.gauss(0,spread),
                    float(depth), float(maxd)])
    end = [pos[i]+dirn[i]*length for i in range(3)]
    if depth < maxd:
        nc = 5 if depth==0 else (4 if depth<3 else 3)
        for _ in range(nc):
            ax = rng.uniform(-0.65,0.65); az = rng.uniform(-0.65,0.65)
            d = dirn[:]
            c2,s2 = math.cos(ax),math.sin(ax)
            d[1],d[2] = d[1]*c2-d[2]*s2, d[1]*s2+d[2]*c2
            c2,s2 = math.cos(az),math.sin(az)
            d[0],d[1] = d[0]*c2-d[1]*s2, d[0]*s2+d[1]*c2
            n = math.sqrt(sum(v*v for v in d))+1e-9; d=[v/n for v in d]
            _grow(end[:], d, length*rng.uniform(0.60,0.74), depth+1, maxd, pts, rng)

def make_tree(maxd=9, seed=7):
    rng = random.Random(seed); pts = []
    _grow([0,-0.93,0],[0,1,0],0.58,0,maxd,pts,rng)
    arr = np.array(pts, dtype=np.float32)
    # sort bottom→top for reveal
    order = np.argsort(arr[:,1])
    return arr[order]

# ── FAST vectorized tree renderer ─────────────────────────────────────────────
def draw_tree_fast(canvas, tree_arr, reveal, ry):
    n = int(len(tree_arr)*min(reveal,1.0))
    if n <= 0: return

    pts  = tree_arr[:n]
    xy   = proj_batch(pts[:,:3], ry)          # (N,2)

    # clip to canvas
    mask = (xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<VIZ_H)
    xy   = xy[mask]; pts = pts[mask]
    if len(xy)==0: return

    depth = pts[:,3]; maxd = pts[:,4]
    t     = depth / np.maximum(maxd, 1)

    # Color by depth
    # trunk (t<0.20): bright white-yellow
    # branches (0.20-0.50): warm white
    # leaves (>0.50): vivid green
    R = np.where(t<0.50, 255.0, 200.0 - t*150.0).clip(0,255)
    G = np.full(len(t), 255.0)
    B = np.where(t<0.20, 230.0,
        np.where(t<0.50, 230.0 - (t-0.20)/0.30*110.0,
                          80.0 - (t-0.50)/0.50*40.0)).clip(0,255)

    # Write core pixels
    glow = np.zeros((VIZ_H, W, 3), dtype=np.float32)
    px, py = xy[:,0], xy[:,1]
    np.add.at(glow, (py, px, 0), B)
    np.add.at(glow, (py, px, 1), G)
    np.add.at(glow, (py, px, 2), R)

    # Bloom: just blur the glow buffer — much faster than per-pixel loops
    glow = np.clip(glow, 0, 255).astype(np.uint8)
    # multi-pass blur for that "lit from inside" look
    blurred1 = cv2.GaussianBlur(glow, (9,9),  0)
    blurred2 = cv2.GaussianBlur(glow, (21,21), 0)
    blurred3 = cv2.GaussianBlur(glow, (41,41), 0)
    composite = np.clip(
        glow.astype(np.float32)*1.0 +
        blurred1.astype(np.float32)*0.9 +
        blurred2.astype(np.float32)*0.6 +
        blurred3.astype(np.float32)*0.3,
        0, 255).astype(np.uint8)

    canvas[:] = np.clip(canvas.astype(np.uint16) + composite, 0, 255).astype(np.uint8)

# ── Particles ─────────────────────────────────────────────────────────────────
class P:
    __slots__=("pos","vel","color","life","maxlife")
    def __init__(self,pos,vel,color,life):
        self.pos=list(pos);self.vel=list(vel);self.color=color;self.life=self.maxlife=float(life)
    def step(self,dt):
        for i in range(3): self.pos[i]+=self.vel[i]*dt
        self.vel[1]-=2.8*dt; self.life-=dt

def spawn_particles(tree_arr):
    ps=[]; step=max(1,len(tree_arr)//400)
    for row in tree_arr[::step]:
        px,py,pz,depth,maxd=row; t=depth/max(maxd,1)
        vel=[random.uniform(-4,4),random.uniform(0,5),random.uniform(-4,4)]
        R=255 if t<0.5 else int(60+100*(1-t)); G=255; B=int(200*(1-t)+40*t)
        ps.append(P([px,py,pz],vel,(B,G,R),random.uniform(2,5)))
    return ps

def draw_particles(canvas,ps,ry):
    for p in ps:
        fade=max(0.0,p.life/p.maxlife); c=tuple(int(v*fade) for v in p.color)
        sx,sy=proj_single(p.pos[0],p.pos[1],p.pos[2],ry); s=max(1,int(fade*7))
        if 0<=sx<W and 0<=sy<VIZ_H: cv2.circle(canvas,(sx,sy),s,c,-1)

# ── Hand drawing ──────────────────────────────────────────────────────────────
TIP_IDS={4,8,12,16,20}

def draw_landmarks(frame,lms,fw,fh,dot_color,label,label_color):
    pts={}
    for i,lm in enumerate(lms.landmark): pts[i]=(int(lm.x*fw),int(lm.y*fh))
    for a,b in CONNECTIONS:
        if a in pts and b in pts: cv2.line(frame,pts[a],pts[b],(180,180,180),1,cv2.LINE_AA)
    for i,(px,py) in pts.items():
        s=9 if i in TIP_IDS else 6
        cv2.rectangle(frame,(px-s,py-s),(px+s,py+s),dot_color if i==0 else (255,255,255),-1)
    wx,wy=pts[0]; lines=label.split("\n"); lw=120; lh=14+len(lines)*16
    lx=max(0,min(wx-lw//2,fw-lw-4)); ly=wy+15
    if ly+lh>fh: ly=wy-lh-15
    ov=frame.copy()
    cv2.rectangle(ov,(lx-4,ly-4),(lx+lw,ly+lh),(0,0,0),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    cv2.rectangle(frame,(lx-4,ly-4),(lx+lw,ly+lh),label_color,2)
    for j,line in enumerate(lines):
        cv2.putText(frame,line,(lx,ly+12+j*16),cv2.FONT_HERSHEY_SIMPLEX,0.45,label_color,1,cv2.LINE_AA)

def get_hands(results):
    left=None; right=None
    if not results.multi_hand_landmarks: return left,right
    for i,lm in enumerate(results.multi_hand_landmarks):
        if i<len(results.multi_handedness):
            lbl=results.multi_handedness[i].classification[0].label
            if lbl=="Right": left=lm
            else: right=lm
    return left,right

# ── UI ────────────────────────────────────────────────────────────────────────
GREEN=(0,220,80);CYAN=(0,220,220);YELLOW=(0,200,255);WHITE=(255,255,255);GRAY=(100,100,100)

def draw_viz_ui(viz, branches, points, height, width, depth, fps):
    items=[("GESTURE CONTROLS",YELLOW,0.50,2),
           ("[LEFT]  Grab / Move Tree",GREEN,0.40,1),
           ("[RIGHT] Add Branches",CYAN,0.40,1),
           ("[BOTH]  Rotate Tree",WHITE,0.40,1),
           ("[SPACE] Reset Tree",GRAY,0.40,1),
           ("[R]     Randomize Tree",GRAY,0.40,1),
           ("[C]     Clear Tree",GRAY,0.40,1)]
    for i,(txt,col,sc,th) in enumerate(items):
        cv2.putText(viz,txt,(10,22+i*21),cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)

    stats=[f"BRANCHES: {branches}",f"POINTS:   {points:,}",
           f"HEIGHT:   {height:.2f} m",f"WIDTH:    {width:.2f} m",
           f"DEPTH:    {depth:.2f} m",f"FPS:      {fps:.0f}"]
    by=VIZ_H-len(stats)*21-8
    for i,s in enumerate(stats):
        cv2.putText(viz,s,(10,by+i*21),cv2.FONT_HERSHEY_SIMPLEX,0.45,GREEN,1,cv2.LINE_AA)

# ── States ────────────────────────────────────────────────────────────────────
S_EMPTY,S_TREE,S_EXPLODE=0,1,2

def main():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cap.isOpened(): print(" No webcam!"); return

    detector=mp_hands.Hands(static_image_mode=False,max_num_hands=2,
        min_detection_confidence=0.6,min_tracking_confidence=0.5)

    state=S_EMPTY; seed=7; maxd=9
    print(" Building tree (one-time)...")
    tree_arr=make_tree(maxd=maxd,seed=seed)
    print(f" Tree ready: {len(tree_arr):,} points")
    tree_reveal=0.0; cube_ry=0.0
    particles=[]; exploded=False
    left_prev_x=left_prev_y=None
    both_prev_x=None
    fps_ring=collections.deque(maxlen=30)
    prev_t=time.time()

    WIN="Advanced Hand Gesture 3D Visualizer"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,W,TOTAL_H)

    print(f" Window: {W}×{TOTAL_H} (portrait)")
    print("  LEFT hand swipe down  → reveal tree")
    print("  LEFT hand L/R         → rotate")
    print("  RIGHT hand            → add branches")
    print("  BOTH spread           →  explode")
    print("  SPACE=reset R=rand C=clear Q=quit\n")

    while True:
        now=time.time(); dt=min(now-prev_t,0.05); prev_t=now
        fps_ring.append(1.0/max(dt,0.001)); fps=sum(fps_ring)/len(fps_ring)

        ret,frame=cap.read()
        if not ret: time.sleep(0.03); continue
        frame=cv2.flip(frame,1)
        frame=cv2.resize(frame,(W,CAM_H))

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=detector.process(rgb)
        left_lm,right_lm=get_hands(results)
        num_hands=(1 if left_lm else 0)+(1 if right_lm else 0)

        key=cv2.waitKey(1)&0xFF
        if key in(ord('q'),ord('Q'),27): break
        elif key==ord(' '):
            state=S_EMPTY;tree_reveal=0.0;maxd=9;seed=7
            tree_arr=make_tree(maxd=maxd,seed=seed);exploded=False
        elif key in(ord('r'),ord('R')):
            seed=random.randint(0,9999)
            tree_arr=make_tree(maxd=maxd,seed=seed)
            tree_reveal=1.0;state=S_TREE
        elif key in(ord('c'),ord('C')): state=S_EMPTY;tree_reveal=0.0

        if left_lm and right_lm:
            mid_x=(left_lm.landmark[0].x+right_lm.landmark[0].x)/2
            if both_prev_x is not None: cube_ry+=(mid_x-both_prev_x)*6.0
            both_prev_x=mid_x; left_prev_x=left_prev_y=None
            spread=math.hypot(left_lm.landmark[0].x-right_lm.landmark[0].x,
                              left_lm.landmark[0].y-right_lm.landmark[0].y)
            if spread>0.38 and not exploded and state==S_TREE:
                state=S_EXPLODE;exploded=True;particles=spawn_particles(tree_arr)
        else:
            both_prev_x=None
            if left_lm and not right_lm:
                lx=left_lm.landmark[0].x; ly=left_lm.landmark[0].y
                if state==S_EMPTY: state=S_TREE;exploded=False
                if left_prev_y is not None:
                    dy=ly-left_prev_y; dx=lx-left_prev_x
                    if dy>0.003: tree_reveal=min(1.0,tree_reveal+dy*5.0)
                    cube_ry+=dx*4.0
                left_prev_x=lx;left_prev_y=ly
            else:
                left_prev_x=left_prev_y=None
                if not left_lm: cube_ry+=dt*0.25
            if right_lm and state==S_TREE:
                new_d=min(maxd+1,11)
                if new_d!=maxd:
                    maxd=new_d
                    print(f" Rebuilding tree depth={maxd}...")
                    tree_arr=make_tree(maxd=maxd,seed=seed)
                    print(f" {len(tree_arr):,} points")

        particles=[p for p in particles if p.life>0]
        for p in particles: p.step(dt)
        if state==S_EXPLODE and not particles:
            state=S_EMPTY;tree_reveal=0.0;exploded=False

        # ── VIZ ───────────────────────────────────────────────────────────
        viz=np.zeros((VIZ_H,W,3),dtype=np.uint8)
        draw_grid_floor(viz,cube_ry)
        draw_cube(viz,cube_ry)

        if state==S_TREE:
            draw_tree_fast(viz,tree_arr,tree_reveal,cube_ry)
        elif state==S_EXPLODE and particles:
            draw_particles(viz,particles,cube_ry)
            blurred=cv2.GaussianBlur(viz,(15,15),0)
            cv2.addWeighted(viz,1.0,blurred,0.5,0,viz)

        branches_n=max(1,int(len(tree_arr)/900))
        points_n=len(tree_arr)
        h_m=9.0+maxd*0.08; w_m=9.8+maxd*0.05; d_m=9.5+maxd*0.05
        draw_viz_ui(viz,branches_n,points_n,h_m,w_m,d_m,fps)

        if state==S_TREE and tree_reveal>0:
            bw=int((W-20)*tree_reveal)
            cv2.rectangle(viz,(10,VIZ_H-4),(10+bw,VIZ_H-1),(0,220,80),-1)

        # Hand landmarks on camera frame
        if results.multi_hand_landmarks:
            for i,lm in enumerate(results.multi_hand_landmarks):
                if i<len(results.multi_handedness):
                    lbl=results.multi_handedness[i].classification[0].label
                    if lbl=="Right":
                        draw_landmarks(frame,lm,W,CAM_H,(0,220,80),"GRAB\nMOVE TREE",(0,220,80))
                    else:
                        draw_landmarks(frame,lm,W,CAM_H,(255,160,0),"ADD\nBRANCHES",(255,160,0))

        # Mode bar
        bar=np.zeros((BAR_H,W,3),dtype=np.uint8); bar[:]=18
        cv2.line(bar,(0,0),(W,0),(55,55,55),1)
        if num_hands==2:  mode="BOTH HANDS | LEFT=MOVE | RIGHT=BRANCHES | BOTH=ROTATE"
        elif left_lm:     mode="LEFT=REVEAL/ROTATE  |  RIGHT=ADD BRANCHES"
        elif right_lm:    mode="RIGHT=ADD BRANCHES  |  LEFT=REVEAL/ROTATE"
        else:             mode="Raise LEFT hand to grow tree  |  SPACE=reset  R=rand"
        cv2.putText(bar,mode,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.36,WHITE,1,cv2.LINE_AA)

        final=np.vstack([viz,frame,bar])
        cv2.imshow(WIN,final)

    cap.release();detector.close();cv2.destroyAllWindows()
    print("Bye!")

if __name__=="__main__":
    main()