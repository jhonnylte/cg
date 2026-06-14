"""
Renderizador 3D de Peças de Xadrez - Versão Simplificada
Phong Shading + Rasterização Baricêntrica + Backface Culling
"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from PIL import Image, ImageTk


# ============================================================
# CLASSE PRINCIPAL DO OBJETO 3D
# ============================================================

class Objeto3D:
    def __init__(self, vertices, faces, normais=None):
        self.vertices = vertices
        self.faces = faces
        self.normais = normais if normais is not None else self._calcular_normais()

    def _calcular_normais(self):
        normais = []
        for f in self.faces:
            AB = self.vertices[f[1]] - self.vertices[f[0]]
            AC = self.vertices[f[2]] - self.vertices[f[0]]
            N = np.cross(AB, AC)
            n = np.linalg.norm(N)
            normais.append(N / n if n > 0 else N)
        return np.array(normais)

    def _hom(self):
        """Adiciona coluna de 1s (coordenadas homogêneas)."""
        return np.hstack([self.vertices, np.ones((len(self.vertices), 1))])

    def translacao(self, tx, ty, tz):
        T = np.eye(4); T[:3, 3] = [tx, ty, tz]
        v = (T @ self._hom().T).T[:, :3]
        return Objeto3D(v, self.faces)

    def escala(self, sx, sy, sz):
        S = np.diag([sx, sy, sz, 1])
        v = (S @ self._hom().T).T[:, :3]
        return Objeto3D(v, self.faces)

    def para_scc(self, view):
        """Transforma vértices e normais para o Sistema de Coordenadas da Câmera."""
        v = (view @ self._hom().T).T[:, :3]
        R = view[:3, :3]
        n = (R @ self.normais.T).T
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return Objeto3D(v, self.faces, n / norms)


# ============================================================
# GEOMETRIAS DAS PEÇAS (via Marching Cubes)
# ============================================================

def _mc(volume, x, y, z):
    """Aplica Marching Cubes e devolve um Objeto3D."""
    verts, faces, norms, _ = marching_cubes(
        volume, level=0,
        spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
    )
    verts[:, 0] += x.min()
    verts[:, 1] += y.min()
    verts[:, 2] += z.min()
    return Objeto3D(verts, faces, norms)


def criar_peao():
    N = 8
    x = np.linspace(-4, 4, N); y = np.linspace(-1, 12, N); z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cilindro = np.maximum(np.sqrt(X**2+Z**2)-3, np.maximum(-Y, Y-2))
    raio_t = 2 - (Y-4)/4
    tronco = np.sqrt(X**2+Z**2) - raio_t; tronco[(Y<2)|(Y>8)] = 1
    esfera = np.sqrt(X**2+(Y-9.5)**2+Z**2) - 2
    return _mc(np.minimum(np.minimum(cilindro, tronco), esfera), x, y, z)


def criar_torre():
    N = 15
    x = np.linspace(-4, 4, N); y = np.linspace(-4, 30, N); z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    base  = np.maximum(np.sqrt(X**2+Z**2)-2.5, np.maximum(-Y,    Y-3))
    corpo = np.maximum(np.sqrt(X**2+Z**2)-2.0, np.maximum(3-Y,   Y-14))
    topo  = np.maximum(np.sqrt(X**2+Z**2)-2.5, np.maximum(14-Y,  Y-16))
    dentes = [
        np.maximum(np.maximum(np.abs(X)-0.5, np.abs(Z-2.2)-0.4), np.maximum(16-Y, Y-19)),
        np.maximum(np.maximum(np.abs(X)-0.5, np.abs(Z+2.2)-0.4), np.maximum(16-Y, Y-19)),
        np.maximum(np.maximum(np.abs(X+2.2)-0.4, np.abs(Z)-0.5), np.maximum(16-Y, Y-19)),
        np.maximum(np.maximum(np.abs(X-2.2)-0.4, np.abs(Z)-0.5), np.maximum(16-Y, Y-19)),
    ]
    vol = np.minimum(np.minimum(base, corpo), topo)
    for d in dentes: vol = np.minimum(vol, d)
    return _mc(vol, x, y, z)


def criar_rei():
    N = 24
    x = np.linspace(-4, 4, N); y = np.linspace(-4, 30, N); z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    base = np.maximum(np.sqrt(X**2+Z**2)-2.5, np.maximum(-Y, Y-3))
    r1 = 2.0 - 0.6*(Y-3)/6;  t1 = np.sqrt(X**2+Z**2)-r1; t1[(Y<3)|(Y>9)] = 1
    r2 = 1.4 + 0.6*(Y-9)/6;  t2 = np.sqrt(X**2+Z**2)-r2; t2[(Y<9)|(Y>15)] = 1
    haste = np.maximum(np.maximum(np.abs(X)-0.4, np.abs(Z)-0.4), np.maximum(15-Y, Y-20))
    barra = np.maximum(np.maximum(np.abs(X)-1.5, np.abs(Z)-0.4), np.maximum(18-Y, Y-19))
    vol = np.minimum(np.minimum(np.minimum(base, t1), t2), np.minimum(haste, barra))
    return _mc(vol, x, y, z)


def criar_rainha():
    N = 24
    x = np.linspace(-4, 4, N); y = np.linspace(-4, 30, N); z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    base = np.maximum(np.sqrt(X**2+Z**2)-2.5, np.maximum(-Y, Y-3))
    r1 = 2.0 - (Y-3)/7;  t1 = np.sqrt(X**2+Z**2)-r1; t1[(Y<3)|(Y>10)] = 1
    r2 = 1.0 + 0.8*(Y-10)/6; t2 = np.sqrt(X**2+Z**2)-r2; t2[(Y<10)|(Y>16)] = 1
    haste = np.maximum(np.maximum(np.abs(X)-0.35, np.abs(Z)-0.35), np.maximum(16-Y, Y-20))
    esfera = np.sqrt(X**2+(Y-21.5)**2+Z**2) - 1.2
    vol = np.minimum(np.minimum(np.minimum(base, t1), t2), np.minimum(haste, esfera))
    return _mc(vol, x, y, z)


def criar_bispo():
    N = 20
    x = np.linspace(-5, 5, N); y = np.linspace(-2, 25, N); z = np.linspace(-5, 5, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    base = np.maximum(np.sqrt(X**2+Z**2)-2.5, np.maximum(-Y, Y-3))
    r1 = 2.0 - (Y-3)/7;  t1 = np.sqrt(X**2+Z**2)-r1; t1[(Y<3)|(Y>10)] = 1
    r2 = 1.0 + 0.8*(Y-10)/6; t2 = np.sqrt(X**2+Z**2)-r2; t2[(Y<10)|(Y>16)] = 1
    cab = np.minimum(np.sqrt(X**2+(Y-18)**2+Z**2)-1.6, np.sqrt(X**2+(Y-20.2)**2+Z**2)-1.2)
    fenda = np.minimum(Y-20+1.2*X, -(Y-20+1.2*X)-0.4)
    cab = np.maximum(cab, -fenda)
    vol = np.minimum(np.minimum(np.minimum(base, t1), t2), cab)
    return _mc(vol, x, y, z)


def criar_dama():
    N = 8
    x = np.linspace(-4, 4, N); y = np.linspace(0, 8, N); z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cil = np.maximum(np.sqrt(X**2+Z**2)-3, np.maximum(3-Y, Y-7))
    return _mc(cil, x, y, z)


# ============================================================
# ILUMINAÇÃO DE PHONG
# ============================================================

def phong(vertices, normais, luz_mundo, view, mat):
    luz = (view @ np.append(luz_mundo, 1))[:3]
    ka, kd, ks, ns = mat["ka"], mat["kd"], mat["ks"], mat["ns"]
    cor = mat["cor"]
    Ia, Id, Is = np.full(3, 0.6), np.full(3, 0.8), np.ones(3)
    cores = np.zeros_like(vertices)
    for i, (P, N) in enumerate(zip(vertices, normais)):
        N = N / (np.linalg.norm(N) or 1)
        L = luz - P; L /= (np.linalg.norm(L) or 1)
        V = -P;      V /= (np.linalg.norm(V) or 1)
        amb = ka * Ia
        nl = np.dot(N, L)
        dif = kd * Id * nl if nl > 0 else np.zeros(3)
        spec = np.zeros(3)
        if nl > 0:
            R = 2*nl*N - L; R /= (np.linalg.norm(R) or 1)
            rv = np.dot(R, V)
            if rv > 0: spec = ks * Is * rv**ns
        cores[i] = np.clip((amb + dif)*cor + spec, 0, 1)
    return cores


# ============================================================
# PROJEÇÃO E RASTERIZAÇÃO
# ============================================================

def projetar(v, f=1, far=None, espelhar=False):
    x, y, z = v
    if z >= 0 or (far and z < -far):
        return None
    sx = 1 if espelhar else -1   # espelha só o eixo horizontal
    return (sx*f*x/z, -f*y/z)   # Y permanece normal


def baricentricas(px, py, A, B, C):
    d = (B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1])
    if abs(d) < 1e-6: return -1, -1, -1
    a = ((B[1]-C[1])*(px-C[0]) + (C[0]-B[0])*(py-C[1])) / d
    b = ((C[1]-A[1])*(px-C[0]) + (A[0]-C[0])*(py-C[1])) / d
    return a, b, 1-a-b


def rasterizar(p1, p2, p3, c1, c2, c3, fb, zb, z1, z2, z3):
    """Rasterização com Z-buffer: só escreve o pixel se estiver mais perto."""
    H, W, _ = fb.shape
    xmin = int(max(0, np.floor(min(p1[0],p2[0],p3[0]))))
    xmax = int(min(W-1, np.ceil(max(p1[0],p2[0],p3[0]))))
    ymin = int(max(0, np.floor(min(p1[1],p2[1],p3[1]))))
    ymax = int(min(H-1, np.ceil(max(p1[1],p2[1],p3[1]))))
    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            a, b, g = baricentricas(x+.5, y+.5, p1, p2, p3)
            if a >= 0 and b >= 0 and g >= 0:
                # Interpola o Z do vértice (valores negativos no SCC; menos negativo = mais perto)
                z_pixel = a*z1 + b*z2 + g*z3
                if z_pixel > zb[y, x]:   # mais perto da câmera → Z menos negativo → maior
                    zb[y, x] = z_pixel
                    fb[y, x] = np.clip(a*c1 + b*c2 + g*c3, 0, 1)


def calcular_view(eye, at, up=np.array([0,1,0])):
    W = eye - at; W /= np.linalg.norm(W)
    U = np.cross(up, W); U /= np.linalg.norm(U)
    V = np.cross(W, U)
    return np.array([
        [U[0], U[1], U[2], -np.dot(U, eye)],
        [V[0], V[1], V[2], -np.dot(V, eye)],
        [W[0], W[1], W[2], -np.dot(W, eye)],
        [0,    0,    0,     1]
    ])


# ============================================================
# CENA
# ============================================================

print("Gerando peças... (aguarde)")

PECAS = [
    {"obj": criar_peao() .translacao( 6,-5,  1).escala(.5,.5,.5),
     "mat": {"ka":.5,"kd":.8,"ks":.5,"ns":32,"cor":np.array([.3,.3,.3])}, "cor3d":"black"},
    {"obj": criar_dama() .translacao(15,-10,15).escala(.5,.3,.5),
     "mat": {"ka":.5,"kd":.8,"ks":.5,"ns":16,"cor":np.array([.9,.2,.2])}, "cor3d":"red"},
    {"obj": criar_rei()  .translacao(-25,-5,-10).escala(.5,.5,.5),
     "mat": {"ka":.5,"kd":.8,"ks":.8,"ns":64,"cor":np.array([.9,.7,.2])}, "cor3d":"goldenrod"},
    {"obj": criar_torre().translacao(-40,-5,-10).escala(.5,.5,.5),
     "mat": {"ka":.5,"kd":.7,"ks":.5,"ns":10,"cor":np.array([.8,.8,.8])}, "cor3d":"silver"},
    {"obj": criar_rainha().translacao(-5,-5,-30).escala(.5,.5,.5),
     "mat": {"ka":.5,"kd":.7,"ks":.5,"ns":10,"cor":np.array([.8,.0,.8])}, "cor3d":"magenta"},
    {"obj": criar_bispo().translacao(-15,-5,-60).escala(.5,.5,.5),
     "mat": {"ka":.5,"kd":.7,"ks":.5,"ns":10,"cor":np.array([.5,.25,.05])}, "cor3d":"saddlebrown"},
]

print("Peças geradas com sucesso!")

estado = {"view": None, "luz": None, "eye_ant": [0., 0., 10.]}


# ============================================================
# RENDERIZAÇÃO
# ============================================================

def renderizar_canvas(view, luz):
    canvas_2d.delete("all")
    W = canvas_2d.winfo_width()
    H = canvas_2d.winfo_height()
    esc = 800
    eye = np.array([float(e.get()) for e in (ent_eye_x, ent_eye_y, ent_eye_z)])
    at  = np.array([float(e.get()) for e in (ent_at_x,  ent_at_y,  ent_at_z)])
    far = np.linalg.norm(eye - at)
    esp = var_espelhar.get()

    # Coleta todas as faces de todos os objetos com seu Z médio
    faces_ordenadas = []
    for p in PECAS:
        obj = p["obj"].para_scc(view)
        cores = phong(obj.vertices, obj.normais, luz, view, p["mat"])
        for face in obj.faces:
            A, B, C = [obj.vertices[face[k]] for k in range(3)]
            if np.cross(B-A, C-A)[2] <= 0: continue
            pts = [projetar(v, far=far, espelhar=esp) for v in (A, B, C)]
            if None in pts: continue
            z_medio = (A[2] + B[2] + C[2]) / 3.0
            cm = sum(cores[face[k]] for k in range(3)) / 3
            hx = "#{:02x}{:02x}{:02x}".format(*[int(c*255) for c in cm])
            xs = [W/2 + pt[0]*esc for pt in pts]
            ys = [H/2 - pt[1]*esc for pt in pts]
            faces_ordenadas.append((z_medio, xs, ys, hx))

    # Painter's algorithm: desenha do mais longe (Z mais negativo) para o mais perto
    faces_ordenadas.sort(key=lambda f: f[0])
    for z_med, xs, ys, hx in faces_ordenadas:
        canvas_2d.create_polygon(xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],
                                 fill=hx, outline="", width=0)


def rasterizar_resolucoes(view, luz):
    eye = np.array([float(e.get()) for e in (ent_eye_x, ent_eye_y, ent_eye_z)])
    at  = np.array([float(e.get()) for e in (ent_at_x,  ent_at_y,  ent_at_z)])
    far = np.linalg.norm(eye - at)
    esp = var_espelhar.get()

    for res in (200, 500, 800):
        fb = np.ones((res, res, 3), dtype=np.float32) * 0.95
        zb = np.full((res, res), -np.inf)          # Z-buffer: começa com -infinito
        esc = res * 0.5
        for p in PECAS:
            obj = p["obj"].para_scc(view)
            cores = phong(obj.vertices, obj.normais, luz, view, p["mat"])
            for face in obj.faces:
                A, B, C = [obj.vertices[face[k]] for k in range(3)]
                if np.cross(B-A, C-A)[2] <= 0: continue
                pts = [projetar(v, far=far, espelhar=esp) for v in (A, B, C)]
                if None in pts: continue
                def sc(pt):
                    return (res/2 + pt[0]*esc, res/2 - pt[1]*esc)
                p1,p2,p3 = [sc(pt) for pt in pts]
                rasterizar(p1, p2, p3,
                           cores[face[0]], cores[face[1]], cores[face[2]], fb, zb,
                           A[2], B[2], C[2])
        img = Image.fromarray((fb*255).astype(np.uint8)).resize((500,500), Image.NEAREST)
        win = tk.Toplevel(janela); win.title(f"Rasterizado {res}×{res}")
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=tk_img); lbl.image = tk_img; lbl.pack()
        img = Image.fromarray((fb*255).astype(np.uint8)).resize((500,500), Image.NEAREST)
        win = tk.Toplevel(janela); win.title(f"Rasterizado {res}×{res}")
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=tk_img); lbl.image = tk_img; lbl.pack()


# ============================================================
# CALLBACKS
# ============================================================

def atualizar():
    try:
        eye = np.array([float(e.get()) for e in (ent_eye_x, ent_eye_y, ent_eye_z)])
        at  = np.array([float(e.get()) for e in (ent_at_x,  ent_at_y,  ent_at_z)])
        luz = np.array([float(e.get()) for e in (ent_luz_x,  ent_luz_y,  ent_luz_z)])
    except ValueError:
        return

    # Manter direção ao mover câmera
    if var_manter.get():
        delta = eye - np.array(estado["eye_ant"])
        if np.any(delta != 0):
            at += delta
            for ent, v in zip((ent_at_x, ent_at_y, ent_at_z), at):
                ent.delete(0, tk.END); ent.insert(0, str(round(v, 2)))

    estado["eye_ant"] = eye.tolist()
    view = calcular_view(eye, at)
    estado["view"] = view
    estado["luz"]  = luz

    # Atualiza gráfico 3D
    ax.cla()
    ax.scatter(*[eye[i] for i in (0,2,1)], c='k', s=80)
    ax.scatter(*[at[i]  for i in (0,2,1)], c='r', s=80)
    ax.scatter(*[luz[i] for i in (0,2,1)], c='y', s=120, marker='*')

    # --- FRUSTUM ---
    # Reordenamos para o sistema do matplotlib (X, Z, Y)
    E = np.array([eye[0], eye[2], eye[1]])
    A = np.array([at[0],  at[2],  at[1]])
    F = A - E
    dist = np.linalg.norm(F)
    if dist > 0:
        f_n = F / dist
        up_g = np.array([0, 0, 1])
        right = np.cross(f_n, up_g)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-3 else np.array([1,0,0])
        up_c  = np.cross(right, f_n)

        # Abertura do frustum proporcional ao canvas 2D
        W_px = canvas_2d.winfo_width()  or 600
        H_px = canvas_2d.winfo_height() or 300
        ESC  = 800
        ax_h = dist * (W_px / 2.0) / ESC   # half-width no plano At
        ay_h = dist * (H_px / 2.0) / ESC   # half-height no plano At

        # 4 cantos do plano far (no At)
        c1 = A + right*ax_h + up_c*ay_h
        c2 = A - right*ax_h + up_c*ay_h
        c3 = A - right*ax_h - up_c*ay_h
        c4 = A + right*ax_h - up_c*ay_h

        # Linhas da câmera até cada canto
        for c in (c1, c2, c3, c4):
            ax.plot([E[0], c[0]], [E[1], c[1]], [E[2], c[2]],
                    color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # Retângulo do plano far
        for a, b in ((c1,c2),(c2,c3),(c3,c4),(c4,c1)):
            ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]],
                    color='gray', linewidth=1.5, alpha=0.8)

    for p in PECAS:
        verts = (p["obj"].para_scc(view) if var_scc.get() else p["obj"]).vertices
        tris = [verts[f][:, [0,2,1]] for f in p["obj"].faces]
        ax.add_collection3d(Poly3DCollection(tris, facecolor=p["cor3d"],
                                              edgecolor=p["cor3d"], linewidth=0.05, alpha=1))

    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.set_box_aspect([1,1,1])
    for lim, setter in zip([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()],
                           [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        c = np.mean(lim); r = max(abs(lim[1]-lim[0]) for lim in
                                   [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]) / 2
        setter([c-r, c+r])
    canvas3d.draw()
    renderizar_canvas(view, luz)


def rasterizar_btn():
    if estado["view"] is None:
        print("Clique em 'Atualizar' primeiro.")
        return
    rasterizar_resolucoes(estado["view"], estado["luz"])


# ============================================================
# INTERFACE
# ============================================================

janela = tk.Tk()
janela.title("Xadrez 3D — Versão Simplificada")
janela.geometry("950x650")

# --- Sidebar ---
sb = tk.Frame(janela, width=260, bg="#e8e8e8", padx=12, pady=12)
sb.pack(side=tk.LEFT, fill=tk.Y)

def campo(parent, label, defaults):
    tk.Label(parent, text=label, font=("Arial",9,"bold"), bg="#e8e8e8").pack(anchor="w", pady=(10,0))
    fr = tk.Frame(parent, bg="#e8e8e8"); fr.pack(fill=tk.X)
    entries = []
    for col, (lbl, val) in enumerate(zip("XYZ", defaults)):
        tk.Label(fr, text=f"{lbl}:", bg="#e8e8e8").grid(row=0, column=col*2)
        e = tk.Entry(fr, width=5); e.insert(0, str(val)); e.grid(row=0, column=col*2+1, padx=2)
        entries.append(e)
    return entries

ent_eye_x, ent_eye_y, ent_eye_z = campo(sb, "👁 Câmera (Eye)", [0, 0, 10])
ent_at_x,  ent_at_y,  ent_at_z  = campo(sb, "🎯 Alvo (At)",    [0, 0, 0])
ent_luz_x, ent_luz_y, ent_luz_z  = campo(sb, "💡 Luz",          [0, 40, 20])

var_manter = tk.BooleanVar(value=True)
tk.Checkbutton(sb, text="Manter direção ao mover", variable=var_manter,
               bg="#e8e8e8").pack(anchor="w", pady=(12,0))

var_scc = tk.BooleanVar(value=False)
tk.Checkbutton(sb, text="Visualizar no SCC", variable=var_scc,
               bg="#e8e8e8").pack(anchor="w")

var_espelhar = tk.BooleanVar(value=False)
tk.Checkbutton(sb, text="🪞 Espelhar projeção", variable=var_espelhar,
               bg="#e8e8e8").pack(anchor="w")

tk.Button(sb, text="▶  Atualizar", command=atualizar,
          bg="#0052cc", fg="white", font=("Arial",10,"bold")).pack(fill=tk.X, pady=18)

tk.Button(sb, text="🖼  Rasterizar (3 res.)", command=rasterizar_btn,
          bg="#cc6600", fg="white", font=("Arial",10,"bold")).pack(fill=tk.X)

# --- Área direita (3D em cima, 2D embaixo) ---
fr_dir = tk.Frame(janela); fr_dir.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
fr_dir.rowconfigure(0, weight=1); fr_dir.rowconfigure(1, weight=1)
fr_dir.columnconfigure(0, weight=1)

fr_3d = tk.Frame(fr_dir); fr_3d.grid(row=0, column=0, sticky="nsew")
fr_2d = tk.Frame(fr_dir); fr_2d.grid(row=1, column=0, sticky="nsew")

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111, projection='3d')
canvas3d = FigureCanvasTkAgg(fig, master=fr_3d)
canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

canvas_2d = tk.Canvas(fr_2d, bg="white")
canvas_2d.pack(fill=tk.BOTH, expand=True)

# Renderização inicial
janela.update()
atualizar()
janela.mainloop()