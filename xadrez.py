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
        return np.hstack([self.vertices, np.ones((len(self.vertices), 1))])

    def translacao(self, tx, ty, tz):
        T = np.eye(4); T[:3, 3] = [tx, ty, tz]
        v = (T @ self._hom().T).T[:, :3]
        return Objeto3D(v, self.faces)

    def escala(self, sx, sy, sz):
        S = np.diag([sx, sy, sz, 1])
        v = (S @ self._hom().T).T[:, :3]
        return Objeto3D(v, self.faces)

    def rotacao_y(self, angulo_graus):
        """Rotação em torno do eixo Y."""
        a = np.radians(angulo_graus)
        R = np.array([
            [ np.cos(a), 0, np.sin(a), 0],
            [         0, 1,         0, 0],
            [-np.sin(a), 0, np.cos(a), 0],
            [         0, 0,         0, 1],
        ])
        v = (R @ self._hom().T).T[:, :3]
        return Objeto3D(v, self.faces)

    def para_scc(self, view):
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


# ============================================================
# PEÇA DE DAMA — cilindro baixo simples (jogo de damas)
# ============================================================

def criar_dama():
    """Cilindro baixo simples, típico de peça do jogo de damas."""
    N = 10
    x = np.linspace(-4, 4, N)
    y = np.linspace(-1, 5, N)
    z = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cil = np.maximum(np.sqrt(X**2 + Z**2) - 3.0, np.maximum(-Y, Y - 4.0))
    return _mc(cil, x, y, z)


# ============================================================
# SUPERFÍCIE — placa quadrada com espessura
# ============================================================

def criar_superficie():
    """
    Placa quadrada plana com pequena espessura, construída diretamente
    como Objeto3D (sem marching cubes, já que é geometria exata).

    A superfície ocupa X ∈ [-10, 10], Z ∈ [-10, 10], Y ∈ [-1.5, 0].
    São 6 faces (12 triângulos) com normais corretas apontando para fora.
    """
    x0, x1 = -10.0,  10.0   # largura
    y0, y1 =  -1.5,   0.0   # espessura (Y)
    z0, z1 = -10.0,  10.0   # profundidade

    # 8 cantos do paralelepípedo
    v = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y0, z1],  # 2
        [x0, y0, z1],  # 3
        [x0, y1, z0],  # 4
        [x1, y1, z0],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ], dtype=float)

    # 12 triângulos (2 por face, normais para fora)
    f = np.array([
        # topo   (Y = y1, normal +Y)
        [4, 6, 5], [4, 7, 6],
        # fundo  (Y = y0, normal -Y)
        [0, 1, 2], [0, 2, 3],
        # frente (Z = z1, normal +Z)
        [3, 6, 7], [3, 2, 6],
        # trás   (Z = z0, normal -Z)
        [0, 5, 1], [0, 4, 5],
        # direita(X = x1, normal +X)
        [1, 5, 6], [1, 6, 2],
        # esquerda(X = x0, normal -X)
        [0, 3, 7], [0, 7, 4],
    ], dtype=int)

    return Objeto3D(v, f)   # normais calculadas automaticamente


# ============================================================
# CORREÇÃO 2: CENA COM VÉRTICES DENTRO DE |10|
# ============================================================

def _normalizar_peca(obj, alvo_max=9.5):
    """
    Translada o objeto para que sua bbox fique centrada na origem e
    reescala uniformemente para que nenhum vértice ultrapasse alvo_max.
    """
    vmin = obj.vertices.min(axis=0)
    vmax = obj.vertices.max(axis=0)
    centro = (vmin + vmax) / 2.0
    obj = obj.translacao(-centro[0], -centro[1], -centro[2])
    escala_max = np.abs(obj.vertices).max()
    if escala_max > 0:
        fator = alvo_max / escala_max
        obj = obj.escala(fator, fator, fator)
    return obj


print("Gerando peças... (aguarde)")

# Gera cada peça normalizada individualmente.
# Restrições para todo vértice final em |10|:
#   - X, Z: alvo_max + |translacao| <= 10  →  alvo_max <= 10 - 3.5 = 6.5
#   - Y: _sobre_superficie move Y de [-am, am] para [0, 2·am]  →  2·am <= 10  →  am <= 5.0
# Logo o limite é alvo_max = 4.9 (margem de 0.1).
_AM = 4.9
_peao_raw   = _normalizar_peca(criar_peao(),   alvo_max=3.5)
_torre_raw  = _normalizar_peca(criar_torre(),  alvo_max=_AM)
_rei_raw    = _normalizar_peca(criar_rei(),    alvo_max=_AM)
_rainha_raw = _normalizar_peca(criar_rainha(), alvo_max=_AM)
_bispo_raw  = _normalizar_peca(criar_bispo(),  alvo_max=_AM)
_dama_raw   = _normalizar_peca(criar_dama(),   alvo_max=1.5)   # cilindro baixo

# Superfície: placa exata, já dentro de |10| por construção
_superficie = criar_superficie()

# Disposição na cena: translações pequenas → todo vértice em |10|
# (max_shape ≤ 6.5, translação ≤ 3.5 → soma ≤ 10)
# As peças ficam sobre a superfície: Y base da superfície = 0,
# então cada peça é transladada em Y pelo valor que coloca seu ponto
# mais baixo em Y = 0.
def _sobre_superficie(obj, tx, tz):
    """Translada a peça para que seu vértice mais baixo fique em Y = 0."""
    y_min = obj.vertices[:, 1].min()
    return obj.translacao(tx, -y_min, tz)

PECAS = [
    {
        "obj": _sobre_superficie(_peao_raw,   0.0,  3.5),
        "mat": {"ka": .5, "kd": .8, "ks": .5, "ns": 32,
                "cor": np.array([.3, .3, .3])},
        "cor3d": "black",
        "nome": "Peão",
    },
    {
        "obj": _sobre_superficie(_dama_raw,   3.5, -3.5),
        "mat": {"ka": .5, "kd": .8, "ks": .5, "ns": 16,
                "cor": np.array([.9, .2, .2])},
        "cor3d": "red",
        "nome": "Dama",
    },
    {
        "obj": _sobre_superficie(_rei_raw,   -3.5,  3.5),
        "mat": {"ka": .5, "kd": .8, "ks": .8, "ns": 64,
                "cor": np.array([.9, .7, .2])},
        "cor3d": "goldenrod",
        "nome": "Rei",
    },
    {
        "obj": _sobre_superficie(_torre_raw, -3.5, -3.5),
        "mat": {"ka": .5, "kd": .7, "ks": .5, "ns": 10,
                "cor": np.array([.8, .8, .8])},
        "cor3d": "silver",
        "nome": "Torre",
    },
    {
        "obj": _sobre_superficie(_rainha_raw, 3.5,  3.5),
        "mat": {"ka": .5, "kd": .7, "ks": .5, "ns": 10,
                "cor": np.array([.8, .0, .8])},
        "cor3d": "magenta",
        "nome": "Rainha",
    },
    {
        "obj": _sobre_superficie(_bispo_raw,  0.0, -3.5),
        "mat": {"ka": .5, "kd": .7, "ks": .5, "ns": 10,
                "cor": np.array([.5, .25, .05])},
        "cor3d": "saddlebrown",
        "nome": "Bispo",
    },
    # Superfície (base)
    {
        "obj": _superficie,
        "mat": {"ka": .4, "kd": .6, "ks": .2, "ns": 8,
                "cor": np.array([.55, .35, .15])},   # madeira clara
        "cor3d": "peru",
        "nome": "Superficie",
    },
]

# Valida que nenhum vértice de peça ultrapassa |10|
for p in PECAS:
    vmax = np.abs(p["obj"].vertices).max()
    assert vmax <= 10.0 + 1e-6, f"ERRO: {p['nome']} tem vértice em {vmax:.2f}"

print("Peças geradas e validadas (todos os vértices em |10|)!")

estado = {"view": None, "luz": None, "eye_ant": [0., 10., 30.]}


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
    sx = 1 if espelhar else -1
    return (sx*f*x/z, -f*y/z)


def calcular_view(eye, at, up=np.array([0, 1, 0])):
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
# CORREÇÃO 4: RASTERIZAÇÃO SCAN-LINE COM PAR-ÍMPAR
# ============================================================

def _interp_cor(t, c0, c1):
    """Interpola linearmente entre duas cores."""
    return c0 + t * (c1 - c0)


def rasterizar_scanline(p1, p2, p3, c1, c2, c3, fb, zb, z1, z2, z3):
    """
    Rasteriza um triângulo usando scan-line com regra par-ímpar.

    Para cada linha de varredura (scan line) y:
      1. Calcula as interseções da linha horizontal com as três arestas.
      2. Ordena as interseções por X.
      3. Preenche os pixels entre o par de interseções (par-ímpar).
      4. Interpola a cor (Gouraud) e o Z linearmente ao longo de cada span.
    """
    H, W, _ = fb.shape

    pts = [p1, p2, p3]
    cols = [c1, c2, c3]
    zs   = [z1, z2, z3]

    # Ordena os vértices por Y crescente
    ordem = sorted(range(3), key=lambda i: pts[i][1])
    A, B, C = [pts[o]   for o in ordem]
    cA, cB, cC = [cols[o] for o in ordem]
    zA, zB, zC = [zs[o]   for o in ordem]

    ymin = int(max(0,   np.ceil (A[1])))
    ymax = int(min(H-1, np.floor(C[1])))

    # Arestas: (início, fim) em termos de índice de vértice
    arestas = [(A, B, cA, cB, zA, zB),
               (B, C, cB, cC, zB, zC),
               (A, C, cA, cC, zA, zC)]

    for y in range(ymin, ymax + 1):
        interseções = []   # lista de (x, cor, z)

        for (va, vb, ca, cb, za_e, zb_e) in arestas:
            ya, yb = va[1], vb[1]
            if ya == yb:
                continue
            # y precisa estar dentro do segmento vertical
            if not (min(ya, yb) <= y <= max(ya, yb)):
                continue
            t = (y - ya) / (yb - ya)
            xi = va[0] + t * (vb[0] - va[0])
            ci = _interp_cor(t, ca, cb)
            zi = za_e + t * (zb_e - za_e)
            interseções.append((xi, ci, zi))

        # Par-ímpar: pares de interseções definem spans
        if len(interseções) < 2:
            continue

        # Ordena por X
        interseções.sort(key=lambda s: s[0])

        # Itera sobre pares (esquerda, direita)
        for k in range(0, len(interseções) - 1, 2):
            xl, cl, zl = interseções[k]
            xr, cr, zr = interseções[k + 1]

            xi_min = int(max(0,   np.ceil (xl)))
            xi_max = int(min(W-1, np.floor(xr)))

            span = xr - xl
            for x in range(xi_min, xi_max + 1):
                t_x = (x - xl) / span if span > 1e-6 else 0.0
                z_px = zl + t_x * (zr - zl)
                if z_px > zb[y, x]:
                    zb[y, x] = z_px
                    fb[y, x] = np.clip(_interp_cor(t_x, cl, cr), 0, 1)


# ============================================================
# RENDERIZAÇÃO 2D (Painter's Algorithm — canvas Tkinter)
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

    faces_ordenadas = []
    for p in PECAS:
        obj = p["obj"].para_scc(view)
        cores = phong(obj.vertices, obj.normais, luz, view, p["mat"])
        for face in obj.faces:
            A, B, C = [obj.vertices[face[k]] for k in range(3)]
            if np.cross(B-A, C-A)[2] <= 0:
                continue
            pts = [projetar(v, far=far, espelhar=esp) for v in (A, B, C)]
            if None in pts:
                continue
            z_medio = (A[2] + B[2] + C[2]) / 3.0
            cm = sum(cores[face[k]] for k in range(3)) / 3
            hx = "#{:02x}{:02x}{:02x}".format(*[int(c*255) for c in cm])
            xs = [W/2 + pt[0]*esc for pt in pts]
            ys = [H/2 - pt[1]*esc for pt in pts]
            faces_ordenadas.append((z_medio, xs, ys, hx))

    faces_ordenadas.sort(key=lambda f: f[0])
    for z_med, xs, ys, hx in faces_ordenadas:
        canvas_2d.create_polygon(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2],
                                 fill=hx, outline="", width=0)


# ============================================================
#  RASTERIZAÇÃO EM 3 RESOLUÇÕES 
# ============================================================

def rasterizar_resolucoes(view, luz):
    eye = np.array([float(e.get()) for e in (ent_eye_x, ent_eye_y, ent_eye_z)])
    at  = np.array([float(e.get()) for e in (ent_at_x,  ent_at_y,  ent_at_z)])
    far = np.linalg.norm(eye - at)
    esp = var_espelhar.get()

    for res in (200, 500, 800):
        fb = np.ones((res, res, 3), dtype=np.float32) * 0.95
        zb = np.full((res, res), -np.inf)
        esc = res * 0.5

        for p in PECAS:
            obj = p["obj"].para_scc(view)
            cores = phong(obj.vertices, obj.normais, luz, view, p["mat"])
            for face in obj.faces:
                A, B, C = [obj.vertices[face[k]] for k in range(3)]
                if np.cross(B-A, C-A)[2] <= 0:
                    continue
                pts = [projetar(v, far=far, espelhar=esp) for v in (A, B, C)]
                if None in pts:
                    continue

                def sc(pt):
                    return (res/2 + pt[0]*esc, res/2 - pt[1]*esc)

                p1, p2, p3 = [sc(pt) for pt in pts]

                # CORREÇÃO 4: usa scan-line em vez de força-bruta
                rasterizar_scanline(
                    p1, p2, p3,
                    cores[face[0]], cores[face[1]], cores[face[2]],
                    fb, zb,
                    A[2], B[2], C[2]
                )

        img = Image.fromarray((fb*255).astype(np.uint8)).resize((500, 500), Image.NEAREST)

        # CORREÇÃO 5: apenas UMA janela por resolução
        win = tk.Toplevel(janela)
        win.title(f"Scan-line rasterizado {res}×{res}")
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=tk_img)
        lbl.image = tk_img   # mantém referência
        lbl.pack()


# ============================================================
# CALLBACKS
# ============================================================

def atualizar():
    try:
        eye = np.array([float(e.get()) for e in (ent_eye_x, ent_eye_y, ent_eye_z)])
        at  = np.array([float(e.get()) for e in (ent_at_x,  ent_at_y,  ent_at_z)])
        luz = np.array([float(e.get()) for e in (ent_luz_x, ent_luz_y, ent_luz_z)])
    except ValueError:
        return

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

    # --- Gráfico 3D ---
    ax.cla()

    # CORREÇÃO 3: origem do SCM marcada explicitamente
    ax.scatter(0, 0, 0, c='lime', s=120, marker='o', zorder=10,
               label='Origem SCM (0,0,0)')
    ax.text(0, 0, 0, '  O (SCM)', fontsize=8, color='green')

    ax.scatter(*[eye[i] for i in (0, 2, 1)], c='k', s=80, label='Eye')
    ax.scatter(*[at[i]  for i in (0, 2, 1)], c='r', s=80, label='At')
    ax.scatter(*[luz[i] for i in (0, 2, 1)], c='y', s=120, marker='*', label='Luz')
    ax.legend(fontsize=7, loc='upper left')

    # Frustum
    E = np.array([eye[0], eye[2], eye[1]])
    A = np.array([at[0],  at[2],  at[1]])
    F = A - E
    dist = np.linalg.norm(F)
    if dist > 0:
        f_n = F / dist
        up_g  = np.array([0, 0, 1])
        right = np.cross(f_n, up_g)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-3 else np.array([1, 0, 0])
        up_c  = np.cross(right, f_n)

        W_px = canvas_2d.winfo_width()  or 600
        H_px = canvas_2d.winfo_height() or 300
        ESC  = 800
        ax_h = dist * (W_px / 2.0) / ESC
        ay_h = dist * (H_px / 2.0) / ESC

        c1 = A + right*ax_h + up_c*ay_h
        c2 = A - right*ax_h + up_c*ay_h
        c3 = A - right*ax_h - up_c*ay_h
        c4 = A + right*ax_h - up_c*ay_h

        for c in (c1, c2, c3, c4):
            ax.plot([E[0], c[0]], [E[1], c[1]], [E[2], c[2]],
                    color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        for a, b in ((c1, c2), (c2, c3), (c3, c4), (c4, c1)):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color='gray', linewidth=1.5, alpha=0.8)

    for p in PECAS:
        verts = (p["obj"].para_scc(view) if var_scc.get() else p["obj"]).vertices
        tris = [verts[f][:, [0, 2, 1]] for f in p["obj"].faces]
        ax.add_collection3d(Poly3DCollection(
            tris, facecolor=p["cor3d"], edgecolor=p["cor3d"],
            linewidth=0.05, alpha=1
        ))

    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.set_box_aspect([1, 1, 1])
    for lim, setter in zip([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()],
                           [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        c = np.mean(lim)
        r = max(abs(lim[1]-lim[0]) for lim in
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
janela.title("Xadrez 3D — Versão Corrigida")
janela.geometry("950x650")

sb = tk.Frame(janela, width=260, bg="#e8e8e8", padx=12, pady=12)
sb.pack(side=tk.LEFT, fill=tk.Y)


def campo(parent, label, defaults):
    tk.Label(parent, text=label, font=("Arial", 9, "bold"), bg="#e8e8e8").pack(anchor="w", pady=(10, 0))
    fr = tk.Frame(parent, bg="#e8e8e8"); fr.pack(fill=tk.X)
    entries = []
    for col, (lbl, val) in enumerate(zip("XYZ", defaults)):
        tk.Label(fr, text=f"{lbl}:", bg="#e8e8e8").grid(row=0, column=col*2)
        e = tk.Entry(fr, width=5); e.insert(0, str(val)); e.grid(row=0, column=col*2+1, padx=2)
        entries.append(e)
    return entries


ent_eye_x, ent_eye_y, ent_eye_z = campo(sb, "👁 Câmera (Eye)", [0, 10, 40])
ent_at_x,  ent_at_y,  ent_at_z  = campo(sb, "🎯 Alvo (At)",    [0,  0,  0])
ent_luz_x, ent_luz_y, ent_luz_z  = campo(sb, "💡 Luz",          [5, 20, 10])

var_manter = tk.BooleanVar(value=True)
tk.Checkbutton(sb, text="Manter direção ao mover", variable=var_manter,
               bg="#e8e8e8").pack(anchor="w", pady=(12, 0))

var_scc = tk.BooleanVar(value=False)
tk.Checkbutton(sb, text="Visualizar no SCC", variable=var_scc,
               bg="#e8e8e8").pack(anchor="w")

var_espelhar = tk.BooleanVar(value=False)
tk.Checkbutton(sb, text="🪞 Espelhar projeção", variable=var_espelhar,
               bg="#e8e8e8").pack(anchor="w")

tk.Button(sb, text="▶  Atualizar", command=atualizar,
          bg="#0052cc", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=18)

tk.Button(sb, text="🖼  Rasterizar (3 res.)", command=rasterizar_btn,
          bg="#cc6600", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X)

# Área direita
fr_dir = tk.Frame(janela); fr_dir.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
fr_dir.rowconfigure(0, weight=1); fr_dir.rowconfigure(1, weight=1)
fr_dir.columnconfigure(0, weight=1)

fr_3d = tk.Frame(fr_dir); fr_3d.grid(row=0, column=0, sticky="nsew")
fr_2d = tk.Frame(fr_dir); fr_2d.grid(row=1, column=0, sticky="nsew")

fig = plt.figure(figsize=(4, 3))
ax  = fig.add_subplot(111, projection='3d')
canvas3d = FigureCanvasTkAgg(fig, master=fr_3d)
canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

canvas_2d = tk.Canvas(fr_2d, bg="white")
canvas_2d.pack(fill=tk.BOTH, expand=True)

janela.update()
atualizar()
janela.mainloop()