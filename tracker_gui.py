"""
DHM Particle Tracker - GUI v3
With 3D Tracking support (VortexLegendre reconstruction + Z search).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, queue, sys, os, cv2, numpy as np
from PIL import Image, ImageTk

AMPLITUDE_FOCUS_METRICS = ['Variance', 'Tenengrad', 'Laplacian']
PHASE_FOCUS_METRICS     = ['Phase Gradient', 'Phase Variance', 'Spectral Energy']

DEFAULTS = {
    'Brightfield': dict(blob_color=0, min_area=400, max_area=5000, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=25, max_skips=5, min_track=5,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True),
    'Amplitude': dict(blob_color=255, min_area=100, max_area=3000, min_circ=0.5,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=20, max_dist=50, max_skips=53, min_track=3,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True),
    'Hologram': dict(blob_color=255, min_area=150, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=5, min_track=5,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True),
    'Phase': dict(blob_color=255, min_area=100, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=True,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=8, min_track=3,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True),
}

MODE_TO_IMAGE = {'Brightfield':'standard','Amplitude':'amplitude',
                 'Hologram':'hologram','Phase':'phase'}


class VideoWindow:
    """Segunda ventana: preview del tracking en tiempo real."""
    def __init__(self, parent, video_w, video_h, scale=0.6, on_stop=None):
        self.win = tk.Toplevel(parent)
        self.win.title('Tracking — Live Preview')
        self.win.resizable(True, True)
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)
        self.closed  = False
        self.on_stop = on_stop

        self.disp_w = max(int(video_w * scale), 640)
        self.disp_h = max(int(video_h * scale), 360)

        # Info bar
        info = tk.Frame(self.win, bg='#2C3E50', pady=4)
        info.pack(fill='x')
        self.info_var = tk.StringVar(value='Starting...')
        tk.Label(info, textvariable=self.info_var, font=('Arial',9),
                 bg='#2C3E50', fg='white').pack(side='left', padx=10)

        # Canvas
        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h,
                                bg='black', highlightthickness=0)
        self.canvas.pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.win, length=self.disp_w, mode='determinate')
        self.progress.pack(fill='x')

        # Control buttons
        btn_frame = tk.Frame(self.win, bg='#ECF0F1', pady=6)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text='⏹  Stop',
            font=('Arial',9,'bold'), bg='#C0392B', fg='white',
            relief='flat', padx=12, pady=4, cursor='hand2',
            command=self._on_stop_click).pack(side='left', padx=10)

        self._photo = None

    # ── Public ───────────────────────────────────────────────────────────────
    def show_frame(self, frame_bgr, fi, total, n_tracks, fps_cur=0.0):
        if self.closed: return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.disp_w, self.disp_h))
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        pct = int(fi / max(total,1) * 100)
        self.progress['value'] = pct
        fps_str = f'  |  {fps_cur:.1f} fps' if fps_cur > 0 else ''
        self.info_var.set(f'Frame {fi}/{total}  |  Tracks: {n_tracks}{fps_str}  |  {pct}%')

    def clear(self):
        """Clears the canvas and resets the progress bar without closing the window."""
        if self.closed: return
        self.canvas.delete('all')
        self._photo = None
        self.progress['value'] = 0
        self.info_var.set('Starting...')

    def close(self):
        if not self.closed:
            self.closed = True
            try: self.win.destroy()
            except: pass

    # ── Internal ─────────────────────────────────────────────────────────────
    def _on_stop_click(self):
        if self.on_stop: self.on_stop()
        self.close()

    def _on_close(self):
        self._on_stop_click()


class TrackerGUI:
    def __init__(self, root):
        self.root      = root
        self.root.title('DHM Particle Tracker')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self._on_quit)

        BG     = '#F4F6F8'
        HDR_BG = '#2C3E50'
        HDR_FG = '#FFFFFF'
        BTN_BG = '#2980B9'
        BTN_FG = '#FFFFFF'

        self.root.configure(bg=BG)
        style = ttk.Style(); style.theme_use('clam')
        for s,cfg in [('TFrame',{'background':BG}),
                      ('TLabel',{'background':BG,'font':('Arial',9)}),
                      ('TEntry',{'font':('Arial',9)}),
                      ('TCheckbutton',{'background':BG,'font':('Arial',9)}),
                      ('TRadiobutton',{'background':BG,'font':('Arial',9)})]:
            style.configure(s, **cfg)

        # Header
        hdr = tk.Frame(root, bg=HDR_BG, pady=10)
        hdr.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(hdr, text='DHM Particle Tracker', font=('Arial',14,'bold'),
                 bg=HDR_BG, fg=HDR_FG).pack()
        tk.Label(hdr, text='Digital Holographic Microscopy Tracking Tool',
                 font=('Arial',9), bg=HDR_BG, fg='#BDC3C7').pack()

        # Scrollable canvas
        cv = tk.Canvas(root, bg=BG, highlightthickness=0, width=520, height=590)
        sb = ttk.Scrollbar(root, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        cv.grid(row=1, column=0, sticky='nsew'); sb.grid(row=1, column=1, sticky='ns')
        root.grid_rowconfigure(1, weight=1); root.grid_columnconfigure(0, weight=1)
        self.main = ttk.Frame(cv, padding=10)
        wid = cv.create_window((0,0), window=self.main, anchor='nw')
        def _cfg(e): cv.configure(scrollregion=cv.bbox('all')); cv.itemconfig(wid, width=cv.winfo_width())
        self.main.bind('<Configure>', _cfg)
        cv.bind('<Configure>', lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all('<MouseWheel>', lambda e: cv.yview_scroll(-1*(e.delta//120),'units'))

        row = 0

        # ── VIDEO TYPE ────────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO TYPE')
        self.mode_var = tk.StringVar(value='Brightfield')
        for m in ['Brightfield','Amplitude','Hologram','Phase']:
            ttk.Radiobutton(self.main, text=m, variable=self.mode_var,
                            value=m, command=self._on_mode).grid(
                row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1); row+=1

        # ── VIDEO FILES ───────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO FILES')
        self.video_path = self._file_row(row, 'Video path:'); row+=1

        # ── OPTICAL SYSTEM ────────────────────────────────────────────────────
        row = self._sec(row, '  OPTICAL SYSTEM')
        self.cam_pixel = self._erow(row, 'Camera pixel size (µm):'); row+=1
        self.magnif    = self._erow(row, 'Magnification (x):');      row+=1
        # FPS: auto-read from video, manual override available
        fps_lbl = ttk.Frame(self.main)
        fps_lbl.grid(row=row, column=0, sticky='w', padx=20, pady=1)
        ttk.Label(fps_lbl, text='FPS:').pack(side='left')
        self.fps_auto_label = ttk.Label(fps_lbl, text='(auto)', foreground='gray')
        self.fps_auto_label.pack(side='left', padx=4)
        fps_ctrl = ttk.Frame(self.main)
        fps_ctrl.grid(row=row, column=1, sticky='w')
        self.fps_override_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fps_ctrl, text='Override:', variable=self.fps_override_var,
                        command=self._toggle_fps_override).pack(side='left')
        self.fps_entry = ttk.Entry(fps_ctrl, width=7, state='disabled')
        self.fps_entry.pack(side='left', padx=4)
        self.fps_entry.insert(0, '15')
        row += 1

        # ── Reconstruction params (used by 3D tracking) ───────────────────────
        ttk.Label(self.main, text='Wavelength λ (µm):').grid(
            row=row, column=0, sticky='w', padx=20, pady=1)
        self.recon_lambda = ttk.Entry(self.main, width=10)
        self.recon_lambda.grid(row=row, column=1, sticky='w', pady=1)
        self.recon_lambda.insert(0, '0.633'); row+=1

        ttk.Label(self.main, text='Vortex filter factor:').grid(
            row=row, column=0, sticky='w', padx=20, pady=1)
        self.recon_factor = ttk.Entry(self.main, width=10)
        self.recon_factor.grid(row=row, column=1, sticky='w', pady=1)
        self.recon_factor.insert(0, '5.0'); row+=1

        # ── DETECTION ─────────────────────────────────────────────────────────
        row = self._sec(row, '  DETECTION')

        # Blob color with enable toggle
        self.filter_color_var = tk.BooleanVar(value=True)
        fc_row = ttk.Frame(self.main); fc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fc_row, text='Filter by blob color:', variable=self.filter_color_var,
                        command=self._toggle_color_filter).pack(side='left')
        self.blob_color_var = tk.IntVar(value=255)
        self.bc_frame = ttk.Frame(fc_row)
        self.bc_frame.pack(side='left', padx=8)
        ttk.Radiobutton(self.bc_frame, text='Dark (0)',     variable=self.blob_color_var, value=0  ).pack(side='left', padx=3)
        ttk.Radiobutton(self.bc_frame, text='Bright (255)', variable=self.blob_color_var, value=255).pack(side='left')
        row+=1

        self.min_area = self._erow(row, 'Min area (px²):');  row+=1
        self.max_area = self._erow(row, 'Max area (px²):');  row+=1

        # Circularity with enable toggle
        self.filter_circ_var = tk.BooleanVar(value=True)
        fcirc_row = ttk.Frame(self.main); fcirc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fcirc_row, text='Filter by circularity  min:', variable=self.filter_circ_var,
                        command=self._toggle_circ_filter).pack(side='left')
        self.min_circ = ttk.Entry(fcirc_row, width=7); self.min_circ.pack(side='left', padx=4)
        row+=1

        ttk.Label(self.main, text='Preprocessing filter:').grid(row=row, column=0, sticky='w', padx=20)
        self.filter_var = tk.StringVar(value='bilateral')
        ff = ttk.Frame(self.main); ff.grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(ff, text='Bilateral', variable=self.filter_var, value='bilateral').pack(side='left', padx=5)
        ttk.Radiobutton(ff, text='Gaussian',  variable=self.filter_var, value='gaussian' ).pack(side='left')
        row+=1

        self.clahe_clip = self._erow(row, 'CLAHE clip limit:'); row+=1

        self.use_dog_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Use DoG + Top-hat (Phase enhancement)',
                        variable=self.use_dog_var, command=self._toggle_dog).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20, pady=2); row+=1
        self.dog_frame = ttk.Frame(self.main)
        self.dog_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=30); row+=1
        ttk.Label(self.dog_frame, text='DoG sigma 1:').grid(row=0,column=0,sticky='w',padx=5)
        self.dog_s1 = ttk.Entry(self.dog_frame, width=8); self.dog_s1.grid(row=0,column=1,padx=5)
        ttk.Label(self.dog_frame, text='DoG sigma 2:').grid(row=0,column=2,sticky='w',padx=5)
        self.dog_s2 = ttk.Entry(self.dog_frame, width=8); self.dog_s2.grid(row=0,column=3,padx=5)
        ttk.Label(self.dog_frame, text='Top-hat kernel:').grid(row=1,column=0,sticky='w',padx=5,pady=2)
        self.tophat = ttk.Entry(self.dog_frame, width=8); self.tophat.grid(row=1,column=1,padx=5)
        self.dog_s1.insert(0,'2.0'); self.dog_s2.insert(0,'8.0'); self.tophat.insert(0,'21')
        self._toggle_dog()

        # ── KALMAN ────────────────────────────────────────────────────────────
        row = self._sec(row, '  KALMAN FILTER')
        ttk.Label(self.main, text='P — Initial covariance', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.P_init = self._erow(row,'P init:'); row+=1
        ttk.Label(self.main, text='Q — Process noise (higher = more agile)', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.Q_val  = self._erow(row,'Q value:'); row+=1
        ttk.Label(self.main, text='R — Measurement noise (higher = trust Kalman more)', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.R_val  = self._erow(row,'R value:'); row+=1

        # ── TRACKING ─────────────────────────────────────────────────────────
        row = self._sec(row, '  TRACKING')
        self.max_dist  = self._erow(row,'Max distance (px):');         row+=1
        self.max_skips = self._erow(row,'Max skips (frames):');        row+=1
        self.min_track = self._erow(row,'Min track length (frames):'); row+=1

        # ── OUTPUT ────────────────────────────────────────────────────────────
        row = self._sec(row, '  OUTPUT')
        self.show_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.main, text='Show trajectory plot after tracking',
                        variable=self.show_plot_var).grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        ttk.Label(self.main, text='Plot style:').grid(row=row,column=0,sticky='w',padx=20)
        self.plot_style_var = tk.StringVar(value='markers')
        ps = ttk.Frame(self.main); ps.grid(row=row,column=1,sticky='w')
        ttk.Radiobutton(ps, text='Markers', variable=self.plot_style_var, value='markers').pack(side='left',padx=5)
        ttk.Radiobutton(ps, text='Dots',    variable=self.plot_style_var, value='dots'   ).pack(side='left')
        row+=1

        self.save_csv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save CSV', variable=self.save_csv_var,
                        command=self._toggle_csv).grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.csv_frame = ttk.Frame(self.main)
        self.csv_frame.grid(row=row,column=0,columnspan=2,sticky='ew',padx=20); row+=1
        ttk.Label(self.csv_frame, text='CSV mode:').grid(row=0,column=0,sticky='w')
        self.csv_mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(self.csv_frame, text='Single file', variable=self.csv_mode_var, value='single'   ).grid(row=0,column=1,padx=5)
        ttk.Radiobutton(self.csv_frame, text='Per track',   variable=self.csv_mode_var, value='per_track').grid(row=0,column=2)
        ttk.Label(self.csv_frame, text='CSV path:').grid(row=1,column=0,sticky='w',pady=2)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(self.csv_frame, textvariable=self.csv_path_var, width=28).grid(row=1,column=1,columnspan=2)
        ttk.Button(self.csv_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.csv_path_var,'trajectories.csv')).grid(row=1,column=3)
        self._toggle_csv()

        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save output video',
                        variable=self.save_video_var, command=self._toggle_video).grid(
            row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.vid_frame = ttk.Frame(self.main)
        self.vid_frame.grid(row=row,column=0,columnspan=2,sticky='ew',padx=20); row+=1
        ttk.Label(self.vid_frame, text='Output video path:').grid(row=0,column=0,sticky='w')
        self.vid_path_var = tk.StringVar()
        ttk.Entry(self.vid_frame, textvariable=self.vid_path_var, width=28).grid(row=0,column=1)
        ttk.Button(self.vid_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.vid_path_var,'output.mp4')).grid(row=0,column=2)
        self._toggle_video()

        # ── 3D TRACKING ───────────────────────────────────────────────────────
        row = self._sec(row, '  3D TRACKING  (VortexLegendre)')

        self.enable_3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Enable 3D Tracking (find Z by refocusing)',
                        variable=self.enable_3d_var,
                        command=self._toggle_3d).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20, pady=2); row+=1

        self._3d_frame = ttk.Frame(self.main)
        self._3d_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=20); row+=1

        # Focus domain
        ttk.Label(self._3d_frame, text='Focus domain:').grid(
            row=0, column=0, sticky='w', padx=5, pady=1)
        self.z_domain_var = tk.StringVar(value='amplitude')
        dom_f = ttk.Frame(self._3d_frame)
        dom_f.grid(row=0, column=1, columnspan=3, sticky='w')
        ttk.Radiobutton(dom_f, text='Amplitude', variable=self.z_domain_var,
                        value='amplitude', command=self._update_metric_list).pack(side='left', padx=4)
        ttk.Radiobutton(dom_f, text='Phase',     variable=self.z_domain_var,
                        value='phase',     command=self._update_metric_list).pack(side='left', padx=4)

        # Focus metric
        ttk.Label(self._3d_frame, text='Focus metric:').grid(
            row=1, column=0, sticky='w', padx=5, pady=1)
        self.z_metric_var = tk.StringVar(value='Tenengrad')
        self.z_metric_cb  = ttk.Combobox(self._3d_frame, textvariable=self.z_metric_var,
                                          width=18, state='readonly')
        self.z_metric_cb.grid(row=1, column=1, columnspan=2, sticky='w', padx=5)
        self._update_metric_list()

        # Z range
        ttk.Label(self._3d_frame, text='Z min (µm):').grid(
            row=2, column=0, sticky='w', padx=5, pady=1)
        self.z_min = ttk.Entry(self._3d_frame, width=9)
        self.z_min.grid(row=2, column=1, sticky='w', padx=5)
        self.z_min.insert(0, '-50')

        ttk.Label(self._3d_frame, text='Z max (µm):').grid(
            row=3, column=0, sticky='w', padx=5, pady=1)
        self.z_max = ttk.Entry(self._3d_frame, width=9)
        self.z_max.grid(row=3, column=1, sticky='w', padx=5)
        self.z_max.insert(0, '50')

        ttk.Label(self._3d_frame, text='Z step (µm):').grid(
            row=4, column=0, sticky='w', padx=5, pady=1)
        self.z_step = ttk.Entry(self._3d_frame, width=9)
        self.z_step.grid(row=4, column=1, sticky='w', padx=5)
        self.z_step.insert(0, '2')

        ttk.Label(self._3d_frame,
                  text='ℹ  All Z values in micrometres (µm)',
                  font=('Arial',8,'italic'), foreground='#2980B9').grid(
            row=5, column=0, columnspan=4, sticky='w', padx=5, pady=(2,4))

        self._toggle_3d()   # start disabled

        # ── BUTTONS BAR ───────────────────────────────────────────────────────
        row+=1
        btn_bar = ttk.Frame(self.main)
        btn_bar.grid(row=row, column=0, columnspan=2, pady=15)

        self.run_btn = tk.Button(btn_bar, text='▶  RUN TRACKING',
            font=('Arial',11,'bold'), bg=BTN_BG, fg=BTN_FG,
            activebackground='#1A6FA8', relief='flat', cursor='hand2',
            padx=20, pady=8, command=self._run)
        self.run_btn.pack(side='left', padx=6)

        tk.Button(btn_bar, text='✕  Exit',
            font=('Arial',11,'bold'), bg='#7F8C8D', fg='white',
            activebackground='#636E72', relief='flat', cursor='hand2',
            padx=16, pady=8, command=self._on_quit).pack(side='left', padx=6)

        # Status bar
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var, font=('Arial',8),
                 bg='#ECF0F1', fg='#555', anchor='w', padx=10).grid(
            row=2, column=0, columnspan=2, sticky='ew')

        # Threading state
        self.frame_queue = queue.Queue(maxsize=4)
        self.video_win   = None
        self._stop_flag  = False
        self._polling    = False
        self._session_id = 0    # incrementa en cada Run — descarta frames de runs viejos

        self._on_mode()

    # ── Widget helpers ────────────────────────────────────────────────────────
    def _sec(self, row, title):
        f = tk.Frame(self.main, bg='#2C3E50', pady=3)
        f.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(8,2))
        tk.Label(f, text=title, font=('Arial',9,'bold'),
                 bg='#2C3E50', fg='white').pack(side='left', padx=8)
        return row+1

    def _erow(self, row, label, width=10):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        e = ttk.Entry(self.main, width=width); e.grid(row=row, column=1, sticky='w', pady=1)
        return e

    def _file_row(self, row, label):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        f = ttk.Frame(self.main); f.grid(row=row, column=1, sticky='w')
        var = tk.StringVar()
        ttk.Entry(f, textvariable=var, width=28).grid(row=0, column=0)
        ttk.Button(f, text='📂', width=3, command=lambda: self._bopen(var)).grid(row=0, column=1)
        return var

    def _bopen(self, var):
        p = filedialog.askopenfilename(
            filetypes=[('Video', '*.avi *.mp4 *.mov *.tif *.tiff'), ('All', '*.*')])
        if p:
            var.set(p)
            if var is self.video_path:
                self._read_video_fps(p)

    def _read_video_fps(self, path):
        """Read FPS from video file and update the info label."""
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self.fps_auto_label.config(text=f'(auto: {fps:.2f})', foreground='#27AE60')
            else:
                self.fps_auto_label.config(text='(auto: N/A)', foreground='gray')
        except Exception:
            self.fps_auto_label.config(text='(auto: error)', foreground='red')

    def _toggle_fps_override(self):
        s = 'normal' if self.fps_override_var.get() else 'disabled'
        self.fps_entry.configure(state=s)

    def _bsave(self, var, default):
        p = filedialog.asksaveasfilename(initialfile=default,
            filetypes=[('CSV','*.csv'),('MP4','*.mp4'),('All','*.*')])
        if p: var.set(p)

    def _set(self, w, v): w.delete(0,'end'); w.insert(0,str(v))

    def _toggle_3d(self):
        state = 'normal' if self.enable_3d_var.get() else 'disabled'
        for child in self._3d_frame.winfo_children():
            try: child.configure(state=state)
            except tk.TclError: pass

    def _update_metric_list(self):
        domain = self.z_domain_var.get()
        if domain == 'amplitude':
            choices, default = AMPLITUDE_FOCUS_METRICS, 'Tenengrad'
        else:
            choices, default = PHASE_FOCUS_METRICS, 'Phase Gradient'
        self.z_metric_cb['values'] = choices
        self.z_metric_var.set(default)

    def _toggle_dog(self):
        s = 'normal' if self.use_dog_var.get() else 'disabled'
        for w in self.dog_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_color_filter(self):
        s = 'normal' if self.filter_color_var.get() else 'disabled'
        for w in self.bc_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_circ_filter(self):
        s = 'normal' if self.filter_circ_var.get() else 'disabled'
        try: self.min_circ.configure(state=s)
        except: pass

    def _toggle_csv(self):
        s = 'normal' if self.save_csv_var.get() else 'disabled'
        for w in self.csv_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_video(self):
        s = 'normal' if self.save_video_var.get() else 'disabled'
        for w in self.vid_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _on_mode(self):
        d = DEFAULTS[self.mode_var.get()]
        self.blob_color_var.set(d['blob_color'])
        self._set(self.min_area, d['min_area']); self._set(self.max_area, d['max_area'])
        self._set(self.min_circ, d['min_circ']); self.filter_var.set(d['filter_type'])
        self._set(self.clahe_clip, d['clahe_clip'])
        self.use_dog_var.set(d['use_dog'])
        self._set(self.dog_s1, d['dog_sigma1']); self._set(self.dog_s2, d['dog_sigma2'])
        self._set(self.tophat, d['tophat_ksize'])
        self._set(self.cam_pixel, d['cam_pixel']); self._set(self.magnif, d['magnification'])
        self._set(self.max_dist, d['max_dist']); self._set(self.max_skips, d['max_skips'])
        self._set(self.min_track, d['min_track'])
        self._set(self.P_init, d['P_init']); self._set(self.Q_val, d['Q_val'])
        self._set(self.R_val, d['R_val'])
        self.filter_color_var.set(d['filter_by_color'])
        self.filter_circ_var.set(d['filter_by_circularity'])
        self._toggle_dog(); self._toggle_color_filter(); self._toggle_circ_filter()

    def _get_fps(self, video_path):
        """Return FPS from video, or override value if enabled."""
        if self.fps_override_var.get():
            try:
                return float(self.fps_entry.get())
            except ValueError:
                raise ValueError('"FPS override" must be a number.')
        fps_auto = 0.0
        try:
            cap = cv2.VideoCapture(video_path)
            fps_auto = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except Exception:
            pass
        return fps_auto if fps_auto > 0 else 15.0

    def _gf(self, w, n):
        try: return float(w.get())
        except: raise ValueError(f'"{n}" must be a number.')

    def _gi(self, w, n):
        try: return int(w.get())
        except: raise ValueError(f'"{n}" must be an integer.')

    # ── Run / Pause / Stop / Quit ─────────────────────────────────────────────
    def _on_quit(self):
        self._stop_flag = True
        if self.video_win: self.video_win.close()
        self.root.destroy()

    def _on_stop(self):
        self._stop_flag = True
        self._polling   = False
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        self.status_var.set('Stopped — ready to restart')
        # Clear the window immediately — don't wait for next poll
        if self.video_win and not self.video_win.closed:
            self.video_win.clear()
        # Drain queue so no stale frames remain
        while True:
            try: self.frame_queue.get_nowait()
            except queue.Empty: break

    def _run(self):
        try:    p = self._collect()
        except ValueError as e: messagebox.showerror('Invalid parameters', str(e)); return

        cap = cv2.VideoCapture(p['video_path'])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 1. Signal the previous thread to stop
        self._stop_flag = True
        self._polling   = False   # cancels the pending _poll immediately

        # 2. Vaciar la queue completamente — desbloquea al thread si estaba llena
        import time as _time
        _drain_start = _time.perf_counter()
        while _time.perf_counter() - _drain_start < 1.0:
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        # second pass to be safe
        while True:
            try: self.frame_queue.get_nowait()
            except queue.Empty: break

        # 3. Limpiar o recrear la ventana de preview
        if self.video_win and not self.video_win.closed:
            self.video_win.clear()   # clear canvas without closing
        else:
            self.video_win = VideoWindow(self.root, w, h, scale=0.6,
                                         on_stop=self._on_stop)

        # 4. Resetear flags y arrancar nuevo thread
        self._stop_flag  = False
        self._polling    = True
        self._session_id += 1          # new session ID — frames from old thread will be discarded
        sid = self._session_id
        self.run_btn.configure(state='disabled', text='⏳  Processing...')
        self.status_var.set('Running tracking...')
        threading.Thread(target=self._thread, args=(p, sid), daemon=True).start()
        self.root.after(30, self._poll)

    def _collect(self):
        v = self.video_path.get()
        if not v: raise ValueError('Select a video file.')
        if not os.path.exists(v): raise ValueError(f'Video not found:\n{v}')

        enable_3d = self.enable_3d_var.get()
        p = {
            'video_path':           v,
            'image_mode':           MODE_TO_IMAGE[self.mode_var.get()],
            'pixel_size':           self._gf(self.cam_pixel,'Camera pixel') / self._gf(self.magnif,'Magnification'),
            'fps':                  self._get_fps(v),
            # optical system (shared with 3D)
            'cam_pixel_um':         self._gf(self.cam_pixel,'Camera pixel size'),
            'recon_lambda':         self._gf(self.recon_lambda,'Wavelength λ'),
            'recon_factor':         self._gf(self.recon_factor,'Vortex filter factor'),
            # detection
            'blob_color':           self.blob_color_var.get(),
            'filter_by_color':      self.filter_color_var.get(),
            'min_area':             self._gi(self.min_area,'Min area'),
            'max_area':             self._gi(self.max_area,'Max area'),
            'filter_by_circ':       self.filter_circ_var.get(),
            'min_circ':             self._gf(self.min_circ,'Min circularity'),
            'filter_type':          self.filter_var.get(),
            'clahe_clip':           self._gf(self.clahe_clip,'CLAHE clip'),
            'use_dog':              self.use_dog_var.get(),
            'dog_sigma1':           self._gf(self.dog_s1,'DoG sigma 1'),
            'dog_sigma2':           self._gf(self.dog_s2,'DoG sigma 2'),
            'tophat_ksize':         self._gi(self.tophat,'Top-hat kernel'),
            'P_init':               self._gf(self.P_init,'P init'),
            'Q_val':                self._gf(self.Q_val,'Q value'),
            'R_val':                self._gf(self.R_val,'R value'),
            'max_dist':             self._gf(self.max_dist,'Max distance'),
            'max_skips':            self._gi(self.max_skips,'Max skips'),
            'min_track':            self._gi(self.min_track,'Min track length'),
            'show_plot':            self.show_plot_var.get(),
            'plot_style':           self.plot_style_var.get(),
            'save_csv':             self.save_csv_var.get(),
            'csv_mode':             self.csv_mode_var.get(),
            'csv_path':             self.csv_path_var.get() or 'trajectories.csv',
            'save_video':           self.save_video_var.get(),
            'video_out':            self.vid_path_var.get() or None,
            # 3D tracking
            'enable_3d':            enable_3d,
            'z_domain':             self.z_domain_var.get(),
            'z_metric':             self.z_metric_var.get(),
            'z_min':                self._gf(self.z_min,  'Z min')  if enable_3d else -50.0,
            'z_max':                self._gf(self.z_max,  'Z max')  if enable_3d else  50.0,
            'z_step':               self._gf(self.z_step, 'Z step') if enable_3d else   2.0,
        }

        if enable_3d and p['z_min'] >= p['z_max']:
            raise ValueError('Z min must be less than Z max.')
        if enable_3d and p['z_step'] <= 0:
            raise ValueError('Z step must be greater than 0.')
        return p

    def _poll(self):
        if not self._polling:
            return
        try:
            while True:
                msg = self.frame_queue.get_nowait()
                # Discard messages from previous runs
                if msg.get('sid') != self._session_id:
                    continue
                if msg['type'] == 'frame':
                    if self.video_win and not self.video_win.closed:
                        self.video_win.show_frame(msg['frame'], msg['fi'],
                                                   msg['total'], msg['n_tracks'],
                                                   msg.get('fps_cur', 0.0))
                elif msg['type'] == 'done':
                    self._polling = False
                    self._on_done(msg); return
                elif msg['type'] == 'error':
                    self._polling = False
                    self._on_error(msg['error']); return
        except queue.Empty: pass
        if not self._stop_flag and self._polling:
            self.root.after(30, self._poll)
        else:
            self._polling = False
            self.run_btn.configure(state='normal', text='▶  RUN TRACKING')

    def _thread(self, p, sid):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from function_tracking_improved import KalmanFilter2D, detect_centroids, _save_csv, _plot
            from scipy.optimize import linear_sum_assignment

            # ── 3D imports (only when needed) ──────────────────────────────────
            if p['enable_3d']:
                import vortexLegendre as VL
                from focus_metrics import compute_focus_metric
                from scipy.fft import fft2, ifft2, fftshift, ifftshift
                from scipy.ndimage import median_filter as _mf

                lambda_  = p['recon_lambda']
                k_wave   = 2 * np.pi / lambda_
                dxy      = p['cam_pixel_um']
                factor   = p['recon_factor']
                z_planes = np.arange(p['z_min'],
                                     p['z_max'] + p['z_step'] / 2,
                                     p['z_step'])

                def _reconstruct(gray):
                    """
                    Full VortexLegendre pipeline:
                    1. Crop to square
                    2. spatial_filter  → holo_filtered
                    3. vortex_compensation → fx_max, fy_max
                    4. reference_wave + multiply → obj_complex
                    5. legendre_compensation → phase_corrected
                    6. Wavefront reconstruction (orders 1-9)
                    7. Final compensation → compensated complex field
                    """
                    sample = gray.astype(np.float64)
                    H_s, W_s = sample.shape
                    if W_s != H_s:
                        lim    = round((W_s - H_s) / 2)
                        sample = sample[:, lim: lim + H_s]

                    N, M   = sample.shape
                    fx_0   = M / 2
                    fy_0   = N / 2
                    m_g, n_g = np.meshgrid(np.arange(-M//2, M//2),
                                           np.arange(-N//2, N//2))

                    # 1-2. Spatial filtering
                    _, holo_filt, fxm, fym, _ = VL.spatial_filter(
                        sample, M, N, save='No', factor=factor, rotate=False)

                    # 3. Vortex compensation
                    logamp = 10 * np.log10(
                        np.abs(fftshift(fft2(fftshift(holo_filt))) + 1e-6)**2)
                    fieldH = _mf(logamp, size=(1, 1), mode='reflect')
                    fxm, fym = VL.vortex_compensation(fieldH, fxm, fym)[0]

                    # 4. Reference wave → complex object field
                    refwa       = VL.reference_wave(fxm, fym, m_g, n_g,
                                                    lambda_, dxy, k_wave,
                                                    fx_0, fy_0, M, N, dy=dxy)
                    obj_complex = refwa * holo_filt

                    # 5. Legendre compensation
                    limit = N / 2
                    _, legendre_coeffs = VL.legendre_compensation(
                        obj_complex, limit,
                        RemovePiston=True, UsePCA=True)

                    # 6. Wavefront reconstruction (orders 1-9)
                    gridSize = obj_complex.shape[0]
                    coords   = np.linspace(-1, 1 - 2/gridSize, gridSize)
                    X_g, Y_g = np.meshgrid(coords, coords)
                    dA       = (2 / gridSize)**2
                    order    = np.arange(1, 10)

                    polynomials = VL.square_legendre_fitting(order, X_g, Y_g)
                    ny, nx, n_polys = polynomials.shape
                    Legendres   = polynomials.reshape(ny * nx, n_polys)
                    zProds      = Legendres.T @ Legendres * dA
                    Legendres   = Legendres / np.sqrt(np.diag(zProds))
                    Legendres_norm_const = np.sum(Legendres**2, axis=0) * dA

                    coeffs    = legendre_coeffs[1:len(order)+1] / np.sqrt(
                                    Legendres_norm_const[:len(order)])
                    coeffs[0] = np.pi + np.pi / 1
                    wavefront = np.sum(
                        coeffs[np.newaxis, :] * Legendres[:, :len(order)], axis=1
                    ).reshape(ny, nx)

                    # 7. Final phase compensation
                    compensated = (np.abs(obj_complex) *
                                   (np.exp(1j * np.angle(obj_complex)) /
                                    np.exp(1j * wavefront)))
                    return compensated

                def _asm_batch(field):
                    N, M = field.shape
                    FX, FY = np.meshgrid(np.fft.fftfreq(M, d=dxy),
                                         np.fft.fftfreq(N, d=dxy))
                    sq = np.sqrt(np.clip(1 - (lambda_*FX)**2 - (lambda_*FY)**2, 0, None))
                    F  = fftshift(fft2(ifftshift(field)))
                    results = []
                    for z in z_planes:
                        H = np.exp(1j * k_wave * z * sq)
                        results.append(fftshift(ifft2(ifftshift(F * H))))
                    return results

                def _find_z(field, detections, frame_shape):
                    """
                    Find optimal Z by evaluating the focus metric ONLY on crops
                    around each 2D-detected particle.
                    Adjusts coordinates to the reconstructed field space (square crop).
                    """
                    propagated = _asm_batch(field)
                    best_z, best_s = z_planes[0], -np.inf

                    fh, fw   = frame_shape[:2]
                    N, M     = field.shape
                    x_off    = (fw - M) // 2
                    y_off    = (fh - N) // 2

                    if len(detections) > 0:
                        crop = 48
                        for z, prop in zip(z_planes, propagated):
                            score   = 0.0
                            n_valid = 0
                            for pt in detections:
                                # Convertir coordenadas del frame al espacio reconstruido
                                cx = int(pt[0]) - x_off
                                cy = int(pt[1]) - y_off
                                x0 = int(np.clip(cx - crop, 0, M))
                                x1 = int(np.clip(cx + crop, 0, M))
                                y0 = int(np.clip(cy - crop, 0, N))
                                y1 = int(np.clip(cy + crop, 0, N))
                                if x1 > x0 and y1 > y0:
                                    patch   = prop[y0:y1, x0:x1]
                                    score  += compute_focus_metric(patch, p['z_domain'],
                                                                   p['z_metric'])
                                    n_valid += 1
                            if n_valid > 0:
                                score /= n_valid
                            if score > best_s:
                                best_s, best_z = score, z
                    else:
                        for z, prop in zip(z_planes, propagated):
                            s = compute_focus_metric(prop, p['z_domain'], p['z_metric'])
                            if s > best_s:
                                best_s, best_z = s, z

                    return best_z

            cap   = cv2.VideoCapture(p['video_path'])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            real_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = real_fps if real_fps > 0 else (p['fps'] if p['fps'] > 0 else 15.0)

            det_kw = dict(image_mode=p['image_mode'],
                          dog_sigma1=p['dog_sigma1'], dog_sigma2=p['dog_sigma2'],
                          clahe_clip=p['clahe_clip'], tophat_ksize=p['tophat_ksize'],
                          use_hough=False)

            # detect always operates on the original gray frame — unchanged in 3D mode
            def detect(gray):
                return detect_centroids(
                    gray, p['min_area'], p['max_area'], p['blob_color'],
                    p['filter_type'], p['filter_by_circ'], p['min_circ'],
                    False, False, p['filter_by_color'], **det_kw)

            ret, first = cap.read()
            if not ret: raise ValueError('Could not read the video.')
            first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

            p0 = detect(first_gray)
            if len(p0) == 0:
                raise ValueError('No particles detected in the first frame.\n'
                                  'Adjust the detection parameters.')

            NAN = np.array([np.nan, np.nan])
            kfs     = [KalmanFilter2D(x, y, p['P_init'], p['Q_val'], p['R_val'])
                       for x, y in p0]
            det_pos = [[p0[i].copy()] for i in range(len(kfs))]
            trajs   = [[kfs[i].state[:2].copy()] for i in range(len(kfs))]
            skips   = [0] * len(kfs)
            done_dp, done_tr = [], []

            # Z history per track (3D mode only)
            if p['enable_3d']:
                z_tracks = [[0.0] for _ in range(len(kfs))]
                done_z   = []
            else:
                z_tracks = None
                done_z   = None

            def _save_t(i):
                real = sum(1 for pt in det_pos[i] if not np.isnan(pt[0]))
                if real >= p['min_track']:
                    done_dp.append([pt.copy() for pt in det_pos[i]])
                    done_tr.append([pt.copy() for pt in trajs[i]])
                    if p['enable_3d']:
                        done_z.append(list(z_tracks[i]))

            writer = None
            if p['save_video'] and p['video_out']:
                h_v, w_v = first.shape[:2]
                writer = cv2.VideoWriter(
                    p['video_out'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_v, h_v))
                if not writer.isOpened():
                    raise ValueError(
                        f"Could not create output video at:\n{p['video_out']}\n\n"
                        "Check that:\n"
                        "  • The destination folder exists\n"
                        "  • You have write permissions\n"
                        "  • The path ends in .mp4")

            mask_2d = np.zeros_like(first)   # trails for 2D mode
            mask_3d = None                    # created when the first recon_field arrives

            # ── Compute Z for frame 0 (if 3D) ──────────────────────────────────
            if p['enable_3d']:
                try:
                    z0 = _find_z(_reconstruct(first_gray))
                    for i in range(len(kfs)):
                        z_tracks[i][0] = z0
                except Exception as ez:
                    print(f'[3D] Frame 0: {ez}')
                    z0 = 0.0
            else:
                z0 = 0.0

            d0 = first.copy()
            for pt in p0:
                cv2.circle(d0, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            if p['enable_3d']:
                cv2.putText(d0, f'Z={z0:.1f}um', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if writer: writer.write(d0)
            try:
                self.frame_queue.put(
                    {'type':'frame','frame':d0,'fi':0,'total':total,'n_tracks':len(kfs),'sid':sid},
                    timeout=0.1)
            except queue.Full:
                pass

            last_recon_field = None   # last valid reconstructed field

            import time
            from collections import deque
            _frame_times = deque(maxlen=10)   # ventana deslizante de 10 frames
            _fps_cur     = 0.0

            for fi in range(1, total):
                if self._stop_flag: break

                _t_frame_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── Z + reconstruction (3D mode only) ─────────────────────────
                # ── 2D detection first (needed for 3D focus crop) ──────────────────
                p1    = detect(gray)
                preds = np.array([kf.predict() for kf in kfs])

                # ── Reconstruction + Z search with crop around particles ───────
                recon_field = None
                if p['enable_3d']:
                    try:
                        recon_field      = _reconstruct(gray)
                        last_recon_field = recon_field
                        z_cur            = _find_z(recon_field, p1, gray.shape)
                    except Exception as ez:
                        print(f'[3D] Frame {fi}: {ez}')
                        recon_field = last_recon_field   # fall back to last valid field
                        z_cur       = 0.0
                else:
                    z_cur = 0.0

                if len(p1) == 0:
                    for i in range(len(kfs)):
                        trajs[i].append(kfs[i].state[:2].copy())
                        det_pos[i].append(NAN.copy()); skips[i] += 1
                        if p['enable_3d']: z_tracks[i].append(z_cur)
                else:
                    D  = np.linalg.norm(preds[:,None,:] - p1[None,:,:], axis=2)
                    ri, ci = linear_sum_assignment(D)
                    ap, ad = set(), set()
                    for r, c in zip(ri, ci):
                        if D[r, c] < p['max_dist']:
                            kfs[r].update(p1[c]); ap.add(r); ad.add(c)
                            det_pos[r].append(p1[c].copy()); skips[r] = 0
                            if p['enable_3d']: z_tracks[r].append(z_cur)
                        else:
                            det_pos[r].append(NAN.copy()); skips[r] += 1
                            if p['enable_3d']: z_tracks[r].append(z_cur)
                    for i in range(len(kfs)):
                        if i not in ap:
                            det_pos[i].append(NAN.copy()); skips[i] += 1
                            if p['enable_3d']: z_tracks[i].append(z_cur)
                    for i in range(len(p1)):
                        if i not in ad:
                            kfs.append(KalmanFilter2D(p1[i][0], p1[i][1],
                                                      p['P_init'], p['Q_val'], p['R_val']))
                            trajs.append([p1[i].copy()])
                            det_pos.append([p1[i].copy()])
                            skips.append(0)
                            if p['enable_3d']: z_tracks.append([z_cur])

                for i, kf in enumerate(kfs): trajs[i].append(kf.state[:2].copy())
                for i in reversed(range(len(kfs))):
                    if skips[i] > p['max_skips']:
                        _save_t(i)
                        del kfs[i], trajs[i], det_pos[i], skips[i]
                        if p['enable_3d']: del z_tracks[i]

                # ── Build display image ─────────────────────────────────────────
                if p['enable_3d'] and recon_field is not None:
                    if p['z_domain'] == 'amplitude':
                        disp_img = np.abs(recon_field).astype(np.float32)
                    else:
                        disp_img = np.angle(recon_field).astype(np.float32)
                    disp_img = cv2.normalize(disp_img, None, 0, 255,
                                             cv2.NORM_MINMAX).astype(np.uint8)
                    disp = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)

                    # Square-crop offset in X (same calculation as _reconstruct)
                    fh, fw = frame.shape[:2]
                    rh, rw = disp.shape[:2]
                    x_off = (fw - rw) // 2   # pixels cropped on the left
                    y_off = (fh - rh) // 2   # pixels cropped on top (normally 0)

                    if mask_3d is None or mask_3d.shape != disp.shape:
                        mask_3d = np.zeros_like(disp)
                    for t in trajs:
                        if len(t) > 1:
                            pts = np.array(t[-2:], dtype=float)
                            # Subtract offset to convert to reconstruction space
                            pts[:, 0] -= x_off
                            pts[:, 1] -= y_off
                            pts = pts.astype(int)
                            # Only draw if within the reconstructed image
                            if (0 <= pts[1,0] < rw and 0 <= pts[1,1] < rh):
                                cv2.line(mask_3d, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                                cv2.circle(disp,  tuple(pts[1]), 4, (0, 0, 255), -1)
                    result = cv2.add(disp, mask_3d)
                elif p['enable_3d'] and recon_field is None:
                    # In 3D mode, never show the raw hologram — show a black frame with a message
                    disp   = np.zeros_like(frame)
                    cv2.putText(disp, 'Reconstruction failed', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    result = disp.copy()
                else:
                    disp = frame.copy()
                    for t in trajs:
                        if len(t) > 1:
                            pts = np.array(t[-2:], dtype=int)
                            cv2.line(mask_2d, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                            cv2.circle(disp,  tuple(pts[1]), 4, (0, 0, 255), -1)
                    result = cv2.add(disp, mask_2d)

                # ── FPS: sliding window ────────────────────────────────────────
                _t_frame_end = time.perf_counter()
                _frame_times.append(_t_frame_end - _t_frame_start)
                _fps_cur = len(_frame_times) / max(sum(_frame_times), 1e-6)

                info = f'Tracks: {len(kfs)}  Frame: {fi}/{total}  {_fps_cur:.1f} fps'
                if p['enable_3d']:
                    info += f'  Z={z_cur:.1f}um'
                cv2.putText(result, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if writer: writer.write(result)
                try:
                    self.frame_queue.put(
                        {'type':'frame','frame':result,'fi':fi,'total':total,
                         'n_tracks':len(kfs),'fps_cur':_fps_cur,'sid':sid},
                        timeout=0.1)
                except queue.Full:
                    pass

            cap.release()
            if writer:
                writer.release()
                print(f'Video saved: {p["video_out"]}')

            if not self._stop_flag:
                for i in range(len(kfs)): _save_t(i)
                self.frame_queue.put({
                    'type':       'done',
                    'det_pos':    done_dp,
                    'trajs':      done_tr,
                    'z_tracks':   done_z,
                    'pixel_size': p['pixel_size'],
                    'n':          len(done_dp),
                    'p':          p,
                    'sid':        sid,
                })

        except Exception as e:
            import traceback
            try:
                if 'writer' in dir() and writer is not None:
                    writer.release()
            except:
                pass
            self.frame_queue.put({'type':'error','error':traceback.format_exc(),'sid':sid})

    def _on_done(self, msg):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        n = msg['n']
        self.status_var.set(f'Done — {n} tracks found.')
        if self.video_win: self.video_win.close()
        if msg['p']['show_plot']:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from function_tracking_improved import _plot
            _plot(msg['det_pos'], msg['trajs'], msg['pixel_size'],
                  style=msg['p']['plot_style'])
            if msg['p']['enable_3d'] and msg.get('z_tracks'):
                self._plot_3d(msg['det_pos'], msg['z_tracks'], msg['pixel_size'])
        if msg['p']['save_csv']:
            from function_tracking_improved import _save_csv
            _save_csv(msg['det_pos'], msg['pixel_size'],
                      msg['p']['csv_mode'], msg['p']['csv_path'])
            if msg['p']['enable_3d'] and msg.get('z_tracks'):
                self._save_z_csv(msg['det_pos'], msg['z_tracks'],
                                 msg['pixel_size'], msg['p']['csv_path'])
        messagebox.showinfo('Tracking complete', f'{n} tracks found.')

    def _plot_3d(self, det_pos, z_tracks, pixel_size):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        n = len(det_pos)
        if n == 0: return
        cmap = plt.cm.tab10

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(n):
            zz = np.array(z_tracks[i])
            ax.plot(np.arange(len(zz)), zz, color=cmap(i % 10),
                    linewidth=1.5, label=f'Track {i}')
        ax.set_xlabel('Frame'); ax.set_ylabel('Z (µm)')
        ax.set_title('Z position over time'); ax.grid(True, alpha=0.4)
        if n <= 20: ax.legend(loc='upper right', fontsize=7)
        plt.tight_layout(); plt.show(block=False)

        fig3d = plt.figure(figsize=(9, 7))
        ax3d  = fig3d.add_subplot(111, projection='3d')
        for i in range(n):
            t  = np.array(det_pos[i], dtype=float) * pixel_size
            zz = np.array(z_tracks[i])
            L  = min(len(t), len(zz))
            t, zz = t[:L], zz[:L]
            valid = ~np.isnan(t[:, 0])
            if valid.sum() < 2: continue
            ax3d.plot(t[valid,0], t[valid,1], zz[valid], color=cmap(i % 10), linewidth=1.5)
            ax3d.scatter(t[valid,0][0],  t[valid,1][0],  zz[valid][0],
                         color=cmap(i%10), s=40, marker='o')
            ax3d.scatter(t[valid,0][-1], t[valid,1][-1], zz[valid][-1],
                         color=cmap(i%10), s=40, marker='s')
        ax3d.set_xlabel('X (µm)'); ax3d.set_ylabel('Y (µm)')
        ax3d.set_zlabel('Z (µm)'); ax3d.set_title('3D Trajectories')
        plt.tight_layout(); plt.show(block=False)

    def _save_z_csv(self, det_pos, z_tracks, pixel_size, base_path):
        import csv
        z_path = base_path.replace('.csv', '_3D.csv')
        with open(z_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['track', 'frame', 'x_um', 'y_um', 'z_um'])
            for i, (traj, zz) in enumerate(zip(det_pos, z_tracks)):
                for fr, (pt, z) in enumerate(zip(traj, zz)):
                    if not np.isnan(pt[0]):
                        w.writerow([i, fr,
                                    f'{pt[0]*pixel_size:.4f}',
                                    f'{pt[1]*pixel_size:.4f}',
                                    f'{z:.4f}'])
        print(f'3D CSV saved: {z_path}')

    def _on_error(self, err):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        self.status_var.set('Error.')
        if self.video_win: self.video_win.close()
        messagebox.showerror('Error', err)


if __name__ == '__main__':
    root = tk.Tk()
    TrackerGUI(root)
    root.mainloop()
