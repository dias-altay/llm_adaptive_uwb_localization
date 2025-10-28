import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import threading, time, queue

def init_plot(anchors_xyz, *, embedded: bool = False, show_raw: bool = True):
    """Create the standard UWB plot (same used by CLI).
    If embedded=True, don't call plt.ion()/plt.show() so it can be embedded in a Qt canvas.
    If show_raw=False, don't create or show the raw position marker.
    """
    if not embedded:
        plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    xs = [p[0] for p in anchors_xyz.values()]
    ys = [p[1] for p in anchors_xyz.values()]
    if xs and ys:
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)

    ax.grid(True, alpha=0.3)

    for aid, pos in anchors_xyz.items():
        ax.scatter([pos[0]], [pos[1]], marker='^')
        ax.text(pos[0], pos[1], f" A{aid}", ha='left', va='bottom')

    tag_dot, = ax.plot([], [], 'ro', label='Tag Position (KF)')
    if show_raw:
        raw_dot, = ax.plot([], [], 'bx', label='Tag Position (Raw)', markersize=10)
    else:
        raw_dot = None

    circles = {}
    for aid, pos in anchors_xyz.items():
        c = mpatches.Circle((pos[0], pos[1]), 0.0, fill=False, linestyle='--', alpha=0.25, linewidth=1.2)
        c.set_visible(False)
        ax.add_patch(c)
        circles[aid] = c

    anchor_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='0.75',
                           markersize=7, linestyle='', label='Anchors')
    handles = [anchor_handle, tag_dot]
    if show_raw and raw_dot is not None:
        handles.append(raw_dot)
    ax.legend(handles=handles, fontsize=8)

    if not embedded:
        plt.show(block=False)
    return fig, ax, raw_dot, tag_dot, circles


def update_plot(fig, raw_dot, tag_dot, circles, x_raw, y_raw, x_kf, y_kf, used_ids, latest):
    if raw_dot is not None:
        raw_dot.set_data([x_raw], [y_raw])
    tag_dot.set_data([x_kf], [y_kf])

    for circ in circles.values():
        circ.set_visible(False)

    for aid_u in used_ids:
        data = latest.get(aid_u)
        if not data:
            continue
        r = data[0]
        c = circles.get(aid_u)
        if c is not None:
            c.set_radius(max(0.0, float(r)))
            c.set_visible(True)

    fig.canvas.draw()
    fig.canvas.flush_events()


def init_plot_with_truth(anchors_xyz, *, embedded: bool = False, show_raw: bool = True):
    fig, ax, raw_dot, tag_dot, circles = init_plot(anchors_xyz, embedded=embedded, show_raw=show_raw)
    gt_dot, = ax.plot([], [], 'go', label='Ground Truth')
    err_line, = ax.plot([], [], linestyle='--', alpha=0.7, linewidth=1.0, color='orange', label='Error')
    info_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=9)
    anchor_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='0.75',
                           markersize=7, linestyle='', label='Anchors')
    handles = [anchor_handle, tag_dot]
    if show_raw and raw_dot is not None:
        handles.append(raw_dot)
    handles.extend([gt_dot, err_line])
    ax.legend(handles=handles, fontsize=8)
    return fig, ax, raw_dot, tag_dot, gt_dot, err_line, circles, info_txt

def update_plot_with_truth(fig, raw_dot, tag_dot, gt_dot, err_line, circles,
                           x_raw, y_raw, x_kf, y_kf, used_ids, latest,
                           gt_x=None, gt_y=None, t=None):
    update_plot(fig, raw_dot, tag_dot, circles, x_raw, y_raw, x_kf, y_kf, used_ids, latest)
    if gt_x is not None and gt_y is not None:
        gt_dot.set_data([gt_x], [gt_y])
        err_line.set_data([gt_x, x_kf], [gt_y, y_kf])
        try:
            de = ((x_kf - gt_x) ** 2 + (y_kf - gt_y) ** 2) ** 0.5
        except Exception:
            de = float('nan')
        ax = fig.axes[0]
        ax.texts[-1].set_text(f"t={t:.3f}\n|e|={de:.3f}" if t is not None else f"|e|={de:.3f}")
    else:
        gt_dot.set_data([], [])
        err_line.set_data([], [])
        ax = fig.axes[0]
        if ax.texts:
            ax.texts[-1].set_text("")
    fig.canvas.draw()
    fig.canvas.flush_events()

def process_pose_packet(fig, artists: dict, pkt: dict, with_truth: bool):
    """
    Unified packet -> artist updater used by CLI, embedded GUI, and any consumer.
    Packet keys:
      x_raw,y_raw,x_kf,y_kf,used_ids,latest,gt_x,gt_y,t
      Optional (for stale GT rendering): gt_stale, gt_last_x, gt_last_y, gt_age_s
    """
    if not pkt:
        return
    x_raw = pkt.get('x_raw'); y_raw = pkt.get('y_raw')
    x_kf = pkt.get('x_kf');   y_kf = pkt.get('y_kf')
    used = pkt.get('used_ids', [])
    latest = pkt.get('latest', {})
    gt_x = pkt.get('gt_x'); gt_y = pkt.get('gt_y'); t = pkt.get('t')
    gt_stale = bool(pkt.get('gt_stale', False))
    gt_last_x = pkt.get('gt_last_x'); gt_last_y = pkt.get('gt_last_y')
    gt_age_s = pkt.get('gt_age_s')

    if with_truth and {'gt_dot','err_line'} <= set(artists):
        update_plot(
            fig,
            artists.get('raw_dot'),
            artists.get('tag_dot'),
            artists.get('circles'),
            x_raw, y_raw, x_kf, y_kf, used, latest
        )

        ax = fig.axes[0]
        info_txt = ax.texts[-1] if ax.texts else None
        gt_dot = artists['gt_dot']
        err_line = artists['err_line']

        if (gt_x is not None) and (gt_y is not None):
            gt_dot.set_data([gt_x], [gt_y])
            try:
                gt_dot.set_alpha(1.0)
            except Exception:
                pass
            err_line.set_data([gt_x, x_kf], [gt_y, y_kf])
            try:
                de = ((x_kf - gt_x) ** 2 + (y_kf - gt_y) ** 2) ** 0.5
            except Exception:
                de = float('nan')
            if info_txt is not None:
                info_txt.set_text(f"t={t:.3f}\n|e|={de:.3f}" if t is not None else f"|e|={de:.3f}")
        elif gt_stale and (gt_last_x is not None) and (gt_last_y is not None):
            gt_dot.set_data([gt_last_x], [gt_last_y])
            try:
                gt_dot.set_alpha(0.25)
            except Exception:
                pass
            err_line.set_data([], [])
            if info_txt is not None:
                if isinstance(gt_age_s, (int, float)):
                    info_txt.set_text(f"GT stale ({gt_age_s:.2f}s)")
                else:
                    info_txt.set_text("GT stale")
        else:
            gt_dot.set_data([], [])
            err_line.set_data([], [])
            if info_txt is not None:
                info_txt.set_text("")
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        update_plot(
            fig,
            artists.get('raw_dot'),
            artists.get('tag_dot'),
            artists.get('circles'),
            x_raw, y_raw, x_kf, y_kf, used, latest
        )

def start_plot_consumer(pose_queue: "queue.Queue", fig, artists: dict, with_truth: bool, stop_event: threading.Event, fps: float = 30.0):
    """
    Launch a background thread that drains pose_queue and updates the figure.
    The producing thread should enqueue packets (process not blocked by matplotlib).
    """
    interval = 1.0 / max(1.0, fps)

    def _run():
        last_draw = 0.0
        while not stop_event.is_set():
            pkt = None
            try:
                pkt = pose_queue.get(timeout=0.05)
                while True:
                    pkt = pose_queue.get_nowait()
            except queue.Empty:
                pass
            if pkt:
                process_pose_packet(fig, artists, pkt, with_truth)
            now = time.time()
            if (now - last_draw) < interval:
                time.sleep(max(0.0, interval - (now - last_draw)))
            last_draw = time.time()

    t = threading.Thread(target=_run, name="PlotConsumer", daemon=True)
    t.start()
    return t
