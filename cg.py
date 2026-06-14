import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageTk


view_matrix_atual = None
posicao_luz_atual = None

def disparar_botao_rasterizar():
    global view_matrix_atual, posicao_luz_atual
    
    # Segurança caso o usuário clique antes de atualizar o gráfico a primeira vez
    if view_matrix_atual is None or posicao_luz_atual is None:
        print("Por favor, clique em 'Atualizar Visualização' primeiro para posicionar a câmera.")
        return

    # Passa diretamente a matriz e a luz que já estavam prontas e calculadas
    executar_rasterizacao(view_matrix_atual, posicao_luz_atual)
    


def calcular_iluminacao_phong(vertices_scc, normais_scc, posicao_luz_mundo, ViewMatrix, propriedades_material):
    """
    Calcula a cor RGB de cada vértice usando o Modelo de Reflexão de Phong.
    Tudo é calculado no Sistema de Coordenadas da Câmera (SCC).
    """
    # 1. Transformar a posição da luz do Mundo para o SCC
    luz_mundo_ch = np.array([posicao_luz_mundo[0], posicao_luz_mundo[1], posicao_luz_mundo[2], 1.0])
    luz_scc = np.dot(ViewMatrix, luz_mundo_ch)[:3]
    
    # Coeficientes do material
    ka = propriedades_material["ka"]
    kd = propriedades_material["kd"]
    ks = propriedades_material["ks"]
    ns = propriedades_material["ns"]
    cor_base = propriedades_material["cor_base"] # np.array([R, G, B]) de 0 a 1
    
    # Intensidades da luz (podem ser brancas [1.0, 1.0, 1.0])
    I_a = np.array([0.6, 0.6, 0.6])
    I_d = np.array([0.8, 0.8, 0.8])
    I_s = np.array([1.0, 1.0, 1.0])
    
    cores_vertices = np.zeros_like(vertices_scc)
    
    # 2. LOOP POR VÉRTICE
    for i in range(len(vertices_scc)):
        P = vertices_scc[i]
        N = normais_scc[i]
        
        # Garantir que a normal está unitária
        norma_n = np.linalg.norm(N)
        if norma_n != 0:
            N = N / norma_n
            
        # Vetor Luz (L)
        L = luz_scc - P
        norma_l = np.linalg.norm(L)
        if norma_l != 0:
            L = L / norma_l
            
        # Vetor Visão (V) -> Câmera está na origem (0,0,0)
        V = -P
        norma_v = np.linalg.norm(V)
        if norma_v != 0:
            V = V / norma_v
            
        # Componente Ambiente
        ambiente = ka * I_a
        
        # Componente Difusa (Lambertiana)
        dot_nl = np.dot(N, L)
        if dot_nl > 0:
            difusa = kd * I_d * dot_nl
        else:
            difusa = np.zeros(3)
            
        # Componente Especular (Phong)
        especular = np.zeros(3)
        if dot_nl > 0:
            # R = 2 * (N . L) * N - L
            R = 2.0 * dot_nl * N - L
            norma_r = np.linalg.norm(R)
            if norma_r != 0:
                R = R / norma_r
                
            dot_rv = np.dot(R, V)
            if dot_rv > 0:
                especular = ks * I_s * (dot_rv ** ns)
                
        # Cor final do vértice = (Ambiente + Difusa) * Cor_do_Objeto + Especular
        # A componente especular geralmente reflete a cor da fonte de luz (brilhante)
        cor_final = (ambiente + difusa) * cor_base + especular
        
        # Clamping para garantir que os valores fiquem entre 0 e 1
        cores_vertices[i] = np.clip(cor_final, 0.0, 1.0)
        
    return cores_vertices


def projetar_ponto(v, f=1):

    x, y, z = v

    if z >= 0:
        return None

    xp = -f * x / z
    yp = -f * y / z

    return xp, yp

def calcular_normais_faces(vertices, faces):

    normais = []

    for face in faces:

        i, j, k = face

        A = vertices[i]
        B = vertices[j]
        C = vertices[k]

        AB = B - A
        AC = C - A

        N = np.cross(AB, AC)

        norma = np.linalg.norm(N)

        if norma != 0:
            N = N / norma

        normais.append(N)

    return np.array(normais)

def calcular_baricentricas(x, y, A, B, C):
    """Calcula as coordenadas baricêntricas de um ponto (x,y) em relação ao triângulo ABC."""
    denominador = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if abs(denominador) < 1e-6:
        return -1, -1, -1  # Triângulo degenerado
        
    alpha = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / denominador
    beta = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / denominador
    gamma = 1.0 - alpha - beta
    
    return alpha, beta, gamma

def rasterizar_triangulo_baricentrico(p1, p2, p3, c1, c2, c3, framebuffer):
    """
    Desenha um único triângulo no framebuffer (matriz NumPy) usando interpolação baricêntrica.
    p1, p2, p3: Coordenadas 2D já mapeadas para a resolução do framebuffer (índices de matriz).
    c1, c2, c3: Cores RGB (0.0 a 1.0) de cada vértice.
    """
    alt, larg, _ = framebuffer.shape
    
    # 1. Determinar os limites da Bounding Box do triângulo no espaço discreto
    xmin = int(max(0, np.floor(min(p1[0], p2[0], p3[0]))))
    xmax = int(min(larg - 1, np.ceil(max(p1[0], p2[0], p3[0]))))
    ymin = int(max(0, np.floor(min(p1[1], p2[1], p3[1]))))
    ymax = int(min(alt - 1, np.ceil(max(p1[1], p2[1], p3[1]))))
    
    # 2. Varrimento por Scanline dentro da região delimitada
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            # Calcular os pesos baricêntricos para o centro do pixel (x + 0.5, y + 0.5)
            alpha, beta, gamma = calcular_baricentricas(x + 0.5, y + 0.5, p1, p2, p3)
            
            # 3. Teste de inclusão (Algoritmo par-ímpar implícito para a geometria do triângulo)
            if alpha >= 0 and beta >= 0 and gamma >= 0:
                # Interpolação linear da cor RGB dos vértices
                cor_pixel = alpha * c1 + beta * c2 + gamma * c3
                
                # Guardar o pixel na matriz (clamping de segurança)
                framebuffer[y, x] = np.clip(cor_pixel, 0.0, 1.0)

def executar_rasterizacao(ViewMatrix, posicao_luz_mundo):
    resolucoes = [200, 500, 800]

    for res in resolucoes:
        framebuffer = np.ones((res, res, 3), dtype=np.float32) * 0.95
        escala_res = res * 0.5 

        # Loop limpo usando a mesma variável global
        for item in objetos_cena:
            obj_scc = item["obj"].transformar_Scc(ViewMatrix)
            cores_vertices = calcular_iluminacao_phong(obj_scc.vertices, obj_scc.normais, posicao_luz_mundo, ViewMatrix, item["mat"])
            
            for idx, face in enumerate(obj_scc.faces):
                A, B, C = obj_scc.vertices[face[0]], obj_scc.vertices[face[1]], obj_scc.vertices[face[2]]
                
                AB = B - A
                AC = C - A
                N_face = np.cross(AB, AC)
                if N_face[2] <= 0:
                    continue
                
                p1, p2, p3 = projetar_ponto(A), projetar_ponto(B), projetar_ponto(C)
                if None in (p1, p2, p3):
                    continue
                
                x1 = res / 2 + p1[0] * escala_res
                y1 = res / 2 - p1[1] * escala_res
                x2 = res / 2 + p2[0] * escala_res
                y2 = res / 2 - p2[1] * escala_res
                x3 = res / 2 + p3[0] * escala_res
                y3 = res / 2 - p3[1] * escala_res
                
                rasterizar_triangulo_baricentrico(
                    (x1, y1), (x2, y2), (x3, y3), 
                    cores_vertices[face[0]], cores_vertices[face[1]], cores_vertices[face[2]], 
                    framebuffer
                )

        # Renderização das janelas Toplevel
        img_uint8 = (framebuffer * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8).resize((500, 500), resample=Image.NEAREST)
        
        janela_res = tk.Toplevel(janela)
        janela_res.title(f"Resolução Real: {res}x{res}")
        img_tk = ImageTk.PhotoImage(img_pil)
        lbl_img = tk.Label(janela_res, image=img_tk)
        lbl_img.image = img_tk  
        lbl_img.pack(expand=True, fill=tk.BOTH)



def adicionar_CH(Po):
      coluna_de_uns = np.ones((Po.shape[0], 1), dtype=int)
      PoCH = np.hstack((Po, coluna_de_uns))
      return PoCH

class Objeto3D:
    def __init__(
        self,
        vertices=None,
        arestas=None,
        faces=None,
        normais=None
    ):
        self.vertices = vertices #or []
        self.arestas = arestas #or []
        self.faces = faces #or []
        self.normais = normais #or []

    def translacao(self, tx, ty, tz):
      T = np.array([
          [1, 0, 0, tx],
          [0, 1, 0, ty],
          [0, 0, 1, tz],
          [0, 0, 0, 1]])
      Pn = np.dot(T, adicionar_CH(self.vertices).T).T
      Pn = Pn[:, :3]
      return Objeto3D(vertices=Pn, faces=self.faces, normais=self.normais)

    def rotacao(self, theta, eixo):
      theta = np.radians(theta)
      if eixo == 'x':
        R = np.array([
            [1, 0, 0,                          0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta),  0],
            [0, 0, 0,                          1]])
      elif eixo == 'y':
        R = np.array([
            [ np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0,                          0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0,                          1]
        ])
      elif eixo == 'z':
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,0,1,0],
            [0,0,0,1]
        ])

      Pn = np.dot(R, adicionar_CH(self.vertices).T).T
      Pn = Pn[:, :3]
      objeto = Objeto3D(vertices=Pn, faces=self.faces, normais=self.normais)
      objeto.atualizar_normais()
      return objeto

    def escala(self, sx, sy, sz):
      S = np.array([
          [sx, 0, 0,  0],
          [0, sy, 0,  0],
          [0, 0,  sz, 0],
          [0, 0,  0,  1]
      ])
      Pn = np.dot(S, adicionar_CH(self.vertices).T).T
      Pn = Pn[:, :3]
      return Objeto3D(vertices=Pn, faces=self.faces, normais=self.normais)

    def transformar_Scc(self, ViewMatrix):
        # Transforma os vértices normalmente (usando coordenadas homogêneas)
        Pn = np.dot(ViewMatrix, adicionar_CH(self.vertices).T).T
        Pn = Pn[:, :3]
        
        # Transforma as normais usando apenas a rotação/escala (submatriz 3x3 superior esquerda)
        # Se houver escala, as normais precisam ser re-normalizadas após a multiplicação
        R_matrix = ViewMatrix[:3, :3]
        Nn = np.dot(R_matrix, self.normais.T).T
        
        # Re-normalizar as normais transformadas
        normas = np.linalg.norm(Nn, axis=1, keepdims=True)
        # Evitar divisão por zero
        normas[normas == 0] = 1.0
        Nn = Nn / normas
        
        objetoScc = Objeto3D(
            vertices=Pn,
            normais=Nn,
            faces=self.faces
        )
        return objetoScc

    def atualizar_normais(self):
      self.normais = calcular_normais_faces(self.vertices, self.faces)


"""========================================================================="""
""" PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO """
"""========================================================================="""

N = 8

x = np.linspace(-4, 4, N)
y = np.linspace(-1, 12, N)
z = np.linspace(-4, 4, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)

# Cilindro peão
lateral = np.sqrt(X**2 + Z**2) - 3

fundo = -Y
topo = Y - 2

cilindro = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

# Tronco de piramide do peão
raio_tronco = 2 - (Y - 4)/4
tronco = np.sqrt(X**2 + Z**2) - raio_tronco
tronco[(Y < 2) | (Y > 8)] = 1

# Esfera da cabeça do peão
esfera = np.sqrt(X**2 + (Y - 9.5)**2 + Z**2) - 2

volume = np.minimum(cilindro, tronco)
volume = np.minimum(volume, esfera)


peao_vertices, peao_faces, peao_normais, _ = marching_cubes(
    volume,
    level=0,
    spacing=(
        x[1]-x[0],
        y[1]-y[0],
        z[1]-z[0]
    )
)

peao_vertices[:,0] += x.min()
peao_vertices[:,1] += y.min()
peao_vertices[:,2] += z.min()


peao = Objeto3D(
    vertices=peao_vertices,
    faces=peao_faces,
    normais=peao_normais
)

peaoScm = peao.translacao(5, -5, 5).escala(0.5, 0.5, 0.5)

"""========================================================================="""
"""==================== PEÃO PEÃO PEÃO PEÃO PEÃO PEÃO ======================"""
"""========================================================================="""



"""========================================================================="""
"""     DAMA DAMA DAMA DAMA DAMA DAMA DAMA DAMA DAMA DAMA DAMA DAMA         """
"""========================================================================="""

N = 8

x = np.linspace(-4, 4, N)
y = np.linspace(0, 8, N)
z = np.linspace(-4, 4, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)

raio = 3
lateral = np.sqrt(X**2 + Z**2) - raio

fundo = 3 - Y
topo = Y - 7

cilindroDama = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)


dama_vertices, dama_faces, dama_normais, _ = marching_cubes(
    cilindroDama,
    level=0,
    spacing=(
        x[1]-x[0],
        y[1]-y[0],
        z[1]-z[0]
    )
)

dama_vertices[:,0] += x.min()
dama_vertices[:,1] += y.min()
dama_vertices[:,2] += z.min()

dama = Objeto3D(
    vertices=dama_vertices,
    faces=dama_faces,
    normais=dama_normais
)

damaScm = dama.translacao(15, -10, 15).escala(0.5, 0.3, 0.5)


"""========================================================================="""
"""==================== DAMA DAMA DAMA DAMA DAMA DAMA ======================"""
"""========================================================================="""




"""========================================================================="""
"""==================== REI REI REI REI REI REI REI REI ===================="""
"""========================================================================="""

N = 24

x = np.linspace(-4, 4, N)
y = np.linspace(-4, 30, N)
z = np.linspace(-4, 4, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)

# -------------------------
# Cilindro base
# -------------------------

raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = -Y
topo = Y - 3

cilindroRei = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

# -------------------------
# Tronco 1
# -------------------------

y0 = 3
y1 = 9

r_base = 2.0
r_topo = 1.4

raio_tronco1 = (
    r_base
    - (r_base - r_topo)
      * (Y - y0)
      / (y1 - y0)
)

tronco1Rei = np.sqrt(X**2 + Z**2) - raio_tronco1
tronco1Rei[(Y < y0) | (Y > y1)] = 1

# -------------------------
# Tronco 2 (invertido)
# -------------------------

y0 = 9
y1 = 15

r_base = 1.4
r_topo = 2.0

raio_tronco2 = (
    r_base
    + (r_topo - r_base)
      * (Y - y0)
      / (y1 - y0)
)

tronco2Rei = np.sqrt(X**2 + Z**2) - raio_tronco2
tronco2Rei[(Y < y0) | (Y > y1)] = 1


# -------------------------
# Cilindro topo
# -------------------------

raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = -Y
topo = Y - 3

cilindroRei = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)


hasteVertical = np.maximum(
    np.abs(X) - 0.4,
    np.maximum(
        np.abs(Z) - 0.4,
        np.maximum(
            15 - Y,
            Y - 20
        )
    )
)

barraHorizontal = np.maximum(
    np.abs(X) - 1.5,
    np.maximum(
        np.abs(Z) - 0.4,
        np.maximum(
            18 - Y,
            Y - 19
        )
    )
)


volume = np.minimum(cilindroRei, tronco1Rei)
volume = np.minimum(volume, tronco2Rei)

volume = np.minimum(volume, hasteVertical)
volume = np.minimum(volume, barraHorizontal)


rei_vertices, rei_faces, rei_normais, _ = marching_cubes(
    volume,
    level=0,
    spacing=(
        x[1]-x[0],
        y[1]-y[0],
        z[1]-z[0]
    )
)

rei_vertices[:,0] += x.min()
rei_vertices[:,1] += y.min()
rei_vertices[:,2] += z.min()

rei = Objeto3D(
    vertices=rei_vertices,
    faces=rei_faces,
    normais=rei_normais
)

reiScm = rei.translacao(-25, -5, -10).escala(0.5, 0.5, 0.5)

"""========================================================================="""
"""==================== REI REI REI REI REI REI REI REI ===================="""
"""========================================================================="""







"""========================================================================="""
"""============= TORRE TORRE TORRE TORRE TORRE TORRE TORRE TORRE ==========="""
"""========================================================================="""

N = 15

x = np.linspace(-4, 4, N)
y = np.linspace(-4, 30, N)
z = np.linspace(-4, 4, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)
raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = -Y
topo = Y - 3

baseTorre = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

raio = 2.0

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = 3 - Y
topo = Y - 14

corpoTorre = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = 14 - Y
topo = Y - 16

topoTorre = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

dente1 = np.maximum(
    np.abs(X) - 0.5,
    np.maximum(
        np.abs(Z - 2.2) - 0.4,
        np.maximum(
            16 - Y,
            Y - 19
        )
    )
)

dente2 = np.maximum(
    np.abs(X) - 0.5,
    np.maximum(
        np.abs(Z + 2.2) - 0.4,
        np.maximum(
            16 - Y,
            Y - 19
        )
    )
)

dente3 = np.maximum(
    np.abs(X + 2.2) - 0.4,
    np.maximum(
        np.abs(Z) - 0.5,
        np.maximum(
            16 - Y,
            Y - 19
        )
    )
)


dente4 = np.maximum(
    np.abs(X - 2.2) - 0.4,
    np.maximum(
        np.abs(Z) - 0.5,
        np.maximum(
            16 - Y,
            Y - 19
        )
    )
)


volume = np.minimum(baseTorre, corpoTorre)
volume = np.minimum(volume, topoTorre)

volume = np.minimum(volume, dente1)
volume = np.minimum(volume, dente2)
volume = np.minimum(volume, dente3)
volume = np.minimum(volume, dente4)

torre_vertices, torre_faces, torre_normais, _ = marching_cubes(
    volume,
    level=0,
    spacing=(
        x[1]-x[0],
        y[1]-y[0],
        z[1]-z[0]
    )
)

torre_vertices[:,0] += x.min()
torre_vertices[:,1] += y.min()
torre_vertices[:,2] += z.min()

torre = Objeto3D(
    vertices=torre_vertices,
    faces=torre_faces,
    normais=torre_normais
)

torreScm = torre.translacao(-40, -5, -10).escala(0.5, 0.5, 0.5)

"""========================================================================="""
"""============= TORRE TORRE TORRE TORRE TORRE TORRE TORRE TORRE ==========="""
"""========================================================================="""



"""========================================================================="""
"""============= RAINHA RAINHA RAINHA RAINHA RAINHA RAINHA RAINHA  ========="""
"""========================================================================="""

# -------------------------
# Base cilíndrica
# -------------------------

raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = -Y
topo = Y - 3

baseRainha = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

# -------------------------
# Tronco 1
# -------------------------

y0 = 3
y1 = 10

r_base = 2.0
r_topo = 1.0

raio_tronco1 = (
    r_base
    - (r_base - r_topo)
      * (Y - y0)
      / (y1 - y0)
)

tronco1Rainha = np.sqrt(X**2 + Z**2) - raio_tronco1
tronco1Rainha[(Y < y0) | (Y > y1)] = 1

# -------------------------
# Tronco 2 (invertido)
# -------------------------

y0 = 10
y1 = 16

r_base = 1.0
r_topo = 1.8

raio_tronco2 = (
    r_base
    + (r_topo - r_base)
      * (Y - y0)
      / (y1 - y0)
)

tronco2Rainha = np.sqrt(X**2 + Z**2) - raio_tronco2
tronco2Rainha[(Y < y0) | (Y > y1)] = 1

# -------------------------
# Haste superior
# -------------------------

hasteRainha = np.maximum(
    np.abs(X) - 0.35,
    np.maximum(
        np.abs(Z) - 0.35,
        np.maximum(
            16 - Y,
            Y - 20
        )
    )
)

# -------------------------
# Esfera superior
# -------------------------

esferaRainha = np.sqrt(
    X**2 +
    (Y - 21.5)**2 +
    Z**2
) - 1.2

# -------------------------
# União dos sólidos
# -------------------------

volume = np.minimum(baseRainha, tronco1Rainha)
volume = np.minimum(volume, tronco2Rainha)
volume = np.minimum(volume, hasteRainha)
volume = np.minimum(volume, esferaRainha)

# -------------------------
# Marching Cubes
# -------------------------

rainha_vertices, rainha_faces, rainha_normais, _ = marching_cubes(
    volume,
    level=0,
    spacing=(
        x[1]-x[0],
        y[1]-y[0],
        z[1]-z[0]
    )
)

rainha_vertices[:,0] += x.min()
rainha_vertices[:,1] += y.min()
rainha_vertices[:,2] += z.min()

rainha = Objeto3D(
    vertices=rainha_vertices,
    faces=rainha_faces,
    normais=rainha_normais
)

rainhaScm = rainha.translacao(-5, -5, -30).escala(0.5, 0.5, 0.5)


"""========================================================================="""
"""============= RAINHA RAINHA RAINHA RAINHA RAINHA RAINHA RAINHA  ========="""
"""========================================================================="""




"""========================================================================="""
"""============= BISPO BISPO BISPO BISPO BISPO BISPO BISPO =========="""
"""========================================================================="""
N = 20

x = np.linspace(-5, 5, N)
y = np.linspace(-2, 25, N)
z = np.linspace(-5, 5, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)
# ==================================================
# BISPO
# ==================================================

# ==================================================
# BISPO
# ==================================================

# -------------------------
# Base cilíndrica
# -------------------------

raio = 2.5

lateral = np.sqrt(X**2 + Z**2) - raio

fundo = -Y
topo = Y - 3

baseBispo = np.maximum(
    lateral,
    np.maximum(fundo, topo)
)

# -------------------------
# Tronco 1
# -------------------------

y0 = 3
y1 = 10

r_base = 2.0
r_topo = 1.0

raio_tronco1 = (
    r_base
    - (r_base - r_topo)
    * (Y - y0)
    / (y1 - y0)
)

tronco1Bispo = np.sqrt(X**2 + Z**2) - raio_tronco1
tronco1Bispo[(Y < y0) | (Y > y1)] = 1

# -------------------------
# Tronco 2 (invertido)
# -------------------------

y0 = 10
y1 = 16

r_base = 1.0
r_topo = 1.8

raio_tronco2 = (
    r_base
    + (r_topo - r_base)
    * (Y - y0)
    / (y1 - y0)
)

tronco2Bispo = np.sqrt(X**2 + Z**2) - raio_tronco2
tronco2Bispo[(Y < y0) | (Y > y1)] = 1

# -------------------------
# Cabeça (duas esferas fundidas)
# -------------------------

esfera1 = np.sqrt(
    X**2 +
    (Y - 18)**2 +
    Z**2
) - 1.6

esfera2 = np.sqrt(
    X**2 +
    (Y - 20.2)**2 +
    Z**2
) - 1.2

cabecaBispo = np.minimum(
    esfera1,
    esfera2
)

# -------------------------
# Fenda diagonal do bispo
# -------------------------

plano1 = Y - 20 + 1.2 * X
plano2 = -(Y - 20 + 1.2 * X) - 0.4

fenda = np.minimum(
    plano1,
    plano2
)

cabecaBispo = np.maximum(
    cabecaBispo,
    -fenda
)

# -------------------------
# Esfera decorativa esquerda
# -------------------------

bola1 = np.sqrt(
    (X + 0.8)**2 +
    (Y - 21.2)**2 +
    Z**2
) - 0.45

# -------------------------
# Esfera decorativa direita
# -------------------------

bola2 = np.sqrt(
    (X - 0.8)**2 +
    (Y - 21.2)**2 +
    Z**2
) - 0.45

# -------------------------
# União dos sólidos
# -------------------------

volume = np.minimum(baseBispo, tronco1Bispo)
volume = np.minimum(volume, tronco2Bispo)
volume = np.minimum(volume, cabecaBispo)
volume = np.minimum(volume, bola1)
volume = np.minimum(volume, bola2)

# -------------------------
# Marching Cubes
# -------------------------

bispo_vertices, bispo_faces, bispo_normais, _ = marching_cubes(
    volume,
    level=0,
    spacing=(
        x[1] - x[0],
        y[1] - y[0],
        z[1] - z[0]
    )
)

bispo_vertices[:, 0] += x.min()
bispo_vertices[:, 1] += y.min()
bispo_vertices[:, 2] += z.min()

bispo = Objeto3D(
    vertices=bispo_vertices,
    faces=bispo_faces,
    normais=bispo_normais
)
bispoScm = bispo.translacao(-15, -5, -60).escala(0.5, 0.5, 0.5)

"""========================================================================="""
"""============= BISPO BISPO BISPO BISPO BISPO BISPO BISPO =========="""
"""========================================================================="""


# Propriedades de materiais mais brilhantes
propriedades_peao  = {"ka": 0.5, "kd": 0.8, "ks": 0.5, "ns": 32, "cor_base": np.array([0.3, 0.3, 0.3])} # Cinza escuro (em vez de preto quase absoluto)
propriedades_dama  = {"ka": 0.5, "kd": 0.8, "ks": 0.5, "ns": 16, "cor_base": np.array([0.9, 0.2, 0.2])} # Vermelho mais vivo
propriedades_rei   = {"ka": 0.5, "kd": 0.8, "ks": 0.8, "ns": 64, "cor_base": np.array([0.9, 0.7, 0.2])} # Dourado vibrante
propriedades_torre = {"ka": 0.5, "kd": 0.7, "ks": 0.5, "ns": 10, "cor_base": np.array([0.8, 0.8, 0.8])} # Prata/Branco
propriedades_rainha = {"ka": 0.5, "kd": 0.7, "ks": 0.5, "ns": 10, "cor_base": np.array([0.8, 0.0, 0.8])} # Magenta
propriedades_bispo = {"ka": 0.5, "kd": 0.7, "ks": 0.5, "ns": 10, "cor_base": np.array([0.5, 0.25, 0.05])} # Prata/Branco
# Mapeamento atualizado no loop
objetos_cena = [
        {"obj": peaoScm, "mat": propriedades_peao, "cor_aresta": "gray"},
        {"obj": damaScm, "mat": propriedades_dama, "cor_aresta": "black"},
        {"obj": reiScm, "mat": propriedades_rei, "cor_aresta": "orange"},
        {"obj": torreScm, "mat": propriedades_torre, "cor_aresta": "dimgray"},
        {"obj": rainhaScm, "mat": propriedades_rainha, "cor_aresta": "purple"}, # Usando prata como base por enquanto
        {"obj": bispoScm, "mat": propriedades_bispo, "cor_aresta": "brown"}
]


""" Variável global para guardar a posição anterior da câmera """
ultimo_eye = [0.0, 0.0, 20.0]

def atualizar_grafico():
    global ultimo_eye, view_matrix_atual, posicao_luz_atual
    
    try:
        ex, ey, ez = float(ent_eye_x.get()), float(ent_eye_y.get()), float(ent_eye_z.get())
        ax_at, ay_at, az_at = float(ent_at_x.get()), float(ent_at_y.get()), float(ent_at_z.get())
        lx, ly, lz = float(ent_luz_x.get()), float(ent_luz_y.get()), float(ent_luz_z.get())
    except ValueError:
        print("Por favor, insira apenas valores numéricos válidos.")
        return

    # ==========================================
    # RECÁLCULO DO ALVO (AT) NO MOVIMENTO
    # ==========================================
    # Calcula o quanto a câmera se moveu desde a última vez
    delta_x = ex - ultimo_eye[0]
    delta_y = ey - ultimo_eye[1]
    delta_z = ez - ultimo_eye[2]
    
    # Se a caixa "Manter Direção" estiver marcada e houve movimento na câmera
    if var_manter_direcao.get() and (delta_x != 0 or delta_y != 0 or delta_z != 0):
        ax_at += delta_x
        ay_at += delta_y
        az_at += delta_z
        
        # Atualiza os campos de texto do Alvo na interface para mostrar os novos valores
        ent_at_x.delete(0, tk.END); ent_at_x.insert(0, str(round(ax_at, 2)))
        ent_at_y.delete(0, tk.END); ent_at_y.insert(0, str(round(ay_at, 2)))
        ent_at_z.delete(0, tk.END); ent_at_z.insert(0, str(round(az_at, 2)))
        
    # Salva a posição atual da câmera para a próxima rodada
    # Sistema de Coordenadas da Câmera
    ultimo_eye = [ex, ey, ez]
    eye = np.array([ex, ey, ez])
    at  = np.array([ax_at, ay_at, az_at])
    up = np.array([0, 1, 0])
    W = eye - at
    W = W / np.linalg.norm(W)
    U = np.cross(up, W)
    U = U / np.linalg.norm(U)
    V = np.cross(W, U)
    ViewMatrix = np.array([
    [U[0], U[1], U[2], -np.dot(U, eye)],
    [V[0], V[1], V[2], -np.dot(V, eye)],
    [W[0], W[1], W[2], -np.dot(W, eye)],
    [0,    0,    0,     1]
    ])

    view_matrix_atual = ViewMatrix
    posicao_luz_mundo = np.array([lx, ly, lz])
    posicao_luz_atual = posicao_luz_mundo 
    peaoScc = peaoScm.transformar_Scc(ViewMatrix)
    damaScc = damaScm.transformar_Scc(ViewMatrix)
    reiScc = reiScm.transformar_Scc(ViewMatrix)
    torreScc = torreScm.transformar_Scc(ViewMatrix)
    rainhaScc = rainhaScm.transformar_Scc(ViewMatrix)
    bispoScc = bispoScm.transformar_Scc(ViewMatrix)


    # =========================================================================
    # LOOP DE RENDERIZAÇÃO 2D COM BACKFACE CULLING (Passos 4 e 6)
    # =========================================================================
    
    # Limpar o canvas 2D antes de desenhar a nova cena atualizada
    canvas_2d.delete("all")
    
    largura = canvas_2d.winfo_width()
    altura = canvas_2d.winfo_height()
    escala = 300  # Fator de escala/zoom para visualização no Canvas 2D

    

    

    for item in objetos_cena:
        # 1. Transformar o objeto do espaço do mundo (SCM) para o espaço da câmera (SCC)
        obj_scc = item["obj"].transformar_Scc(ViewMatrix)
        
        # OBRIGATÓRIO: Recalcular as normais diretamente no SCC para alinhar com os vértices transformados
        obj_scc.atualizar_normais()
        
        # Calcular a cor de todos os vértices do objeto via Phong no SCC
        cores_vertices = calcular_iluminacao_phong(obj_scc.vertices, obj_scc.normais, posicao_luz_mundo, ViewMatrix, item["mat"])

        # 2. Iterar polígono por polígono (face por face) do modelo
        for idx, face in enumerate(obj_scc.faces):
            # Extrair os três vértices associados à face atual no SCC
            A = obj_scc.vertices[face[0]]
            B = obj_scc.vertices[face[1]]
            C = obj_scc.vertices[face[2]]
            
            # Obter o vetor normal da face calculado no espaço da câmera
            N = obj_scc.normais[idx]
            
            # 3. BACKFACE CULLING (Requisito 6)
            # Avalia a visibilidade comparando a normal do polígono com o eixo Z da câmera.
            # No SCC, se a componente Z da normal for maior que zero (N[2] > 0), a face aponta
            # na direção do observador. Caso contrário, está oculta e é descartada.
            # Nota: Dependendo da orientação horária/anti-horária dos triângulos gerados pelo 
            # marching cubes, caso note que as faces visíveis sumiram e as ocultas ficaram, 
            # altere a condição para N[2] >= 0.
            if N[2] <= 0:
                continue  # Ignora a renderização desta face e passa para a próxima
                
            # 4. PROJEÇÃO PERSPETIVA 3D -> 2D (Requisito 4)
            p1 = projetar_ponto(A)
            p2 = projetar_ponto(B)
            p3 = projetar_ponto(C)
            
            # Se algum dos vértices estiver fora do volume de visão traseiro (z >= 0), descarta
            if None in (p1, p2, p3):
                continue
                
            # 5. MAPEAMENTO DE VIEWPORT (Conversão para coordenadas de tela do Canvas)
            # Inverte-se o eixo Y matemático para adequar ao padrão de coordenadas de ecrã (Y cresce para baixo)
            x1 = largura / 2 + p1[0] * escala
            y1 = altura / 2 - p1[1] * escala
            
            x2 = largura / 2 + p2[0] * escala
            y2 = altura / 2 - p2[1] * escala
            
            x3 = largura / 2 + p3[0] * escala
            y3 = altura / 2 - p3[1] * escala
            
            # Encontrar a cor média da face para pintura provisória no Canvas
            cor_media = (cores_vertices[face[0]] + cores_vertices[face[1]] + cores_vertices[face[2]]) / 3.0
            r_hex = int(cor_media[0] * 255)
            g_hex = int(cor_media[1] * 255)
            b_hex = int(cor_media[2] * 255)
            cor_hex = f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"
            
            # 6. RASTERIZAÇÃO PROVISÓRIA VIA CANVAS (A ser substituída pelo algoritmo próprio no Passo 6)
            canvas_2d.create_polygon(
                x1, y1,
                x2, y2,
                x3, y3,
                fill=cor_hex,
                outline="",
                width=0
            )
    # ==========================================
    # DESENHO DO GRÁFICO (CANVAS)
    # ==========================================
    eixo_3d.cla()


    # 2. Desenhar Alvo e Câmera
    eixo_3d.scatter(ax_at, az_at, ay_at, color='red', s=150, marker='.', label='')
    eixo_3d.scatter(ex, ez, ey, color='black', s=100, marker='.', label='')
    eixo_3d.scatter(0, 0, 0, color='green', s=200, marker='.', label='Origem (0,0,0)')
    eixo_3d.scatter(posicao_luz_mundo[0], posicao_luz_mundo[2], posicao_luz_mundo[1], color='yellow', s=200, marker='.', label='Luz')
    """========================================================================="""
    """ Desenhar o PEÃO usando os vértices gerados pelo marching cubes """
    """========================================================================="""

    if var_mostrar_scc.get():
        Vpeao = peaoScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vpeao = peaoScm.vertices

    triangulosPeao = []

    for face in peaoScm.faces:
        triangulo = Vpeao[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosPeao.append(triangulo)

    malhaPeao = Poly3DCollection(
        triangulosPeao,
        edgecolor='black',
        facecolor='black',
        linewidth=0.1,
        alpha=1.0
    )

    eixo_3d.add_collection3d(malhaPeao)
    """========================================================================="""
    """========================================================================="""





    """========================================================================="""
    """ Desenhar a DAMA usando os vértices gerados pelo marching cubes """
    """========================================================================="""
    if var_mostrar_scc.get():
        Vdama = damaScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vdama = damaScm.vertices

    triangulosDama = []

    for face in damaScm.faces:
        triangulo = Vdama[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosDama.append(triangulo)

    malhaDama = Poly3DCollection(
        triangulosDama,
        edgecolor='red',
        facecolor='red',
        linewidth=0.1,
        alpha=1
    )

    eixo_3d.add_collection3d(malhaDama)


    """========================================================================="""
    """========================================================================="""



    """========================================================================="""
    """ Desenhar o REI usando os vértices gerados pelo marching cubes """
    """========================================================================="""
    if var_mostrar_scc.get():
        Vrei = reiScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vrei = reiScm.vertices

    triangulosRei = []

    for face in reiScm.faces:
        triangulo = Vrei[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosRei.append(triangulo)

    malhaRei = Poly3DCollection(
        triangulosRei,
        edgecolor='gold',
        facecolor='goldenrod',
        linewidth=0.1,
        alpha=1
    )

    eixo_3d.add_collection3d(malhaRei)

    """========================================================================="""
    """========================================================================="""




    """========================================================================="""
    """ Desenhar a TORRE usando os vértices gerados pelo marching cubes """
    """========================================================================="""
    if var_mostrar_scc.get():
        Vtorre = torreScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vtorre = torreScm.vertices

    triangulosTorre = []

    for face in torreScm.faces:
        triangulo = Vtorre[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosTorre.append(triangulo)

    malhaTorre = Poly3DCollection(
        triangulosTorre,
        edgecolor='dimgray',
        facecolor='silver',
        linewidth=0.1,
        alpha=1
    )

    eixo_3d.add_collection3d(malhaTorre)
  
  
  
  
  
    """========================================================================="""
    """========================================================================="""


    """========================================================================="""
    """ Desenhar a RAINHA usando os vértices gerados pelo marching cubes """
    """========================================================================="""

    if var_mostrar_scc.get():
        Vrainha = rainhaScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vrainha = rainhaScm.vertices

    triangulosRainha = []
    for face in rainhaScm.faces:
        triangulo = Vrainha[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosRainha.append(triangulo)
    malhaRainha = Poly3DCollection(
        triangulosRainha,
        edgecolor='purple',
        facecolor='magenta',
        linewidth=0.1,
        alpha=1
    )
    eixo_3d.add_collection3d(malhaRainha)


    """========================================================================="""
    """========================================================================="""

    """========================================================================="""
    """ Desenhar o BISPO usando os vértices gerados pelo marching cubes """
    """========================================================================="""

    if var_mostrar_scc.get():
        Vbispo = bispoScm.transformar_Scc(ViewMatrix).vertices
    else:
        Vbispo = bispoScm.vertices

    triangulosBispo = []
    for face in bispoScm.faces:
        triangulo = Vbispo[face]
        triangulo = triangulo[:, [0,2,1]]
        triangulosBispo.append(triangulo)
    malhaBispo = Poly3DCollection(
        triangulosBispo,
        edgecolor='peachpuff',
        facecolor='saddlebrown',
        linewidth=0.1,
        alpha=1
    )
    eixo_3d.add_collection3d(malhaBispo)


    """========================================================================="""
    """========================================================================="""


    # 3. Caixa de Visualização (Frustum)
    E = np.array([ex, ez, ey])      
    A = np.array([ax_at, az_at, ay_at]) 
    F = A - E
    dist = np.linalg.norm(F)
    
    if dist > 0:
        f_norm = F / dist
        up_global = np.array([0, 0, 1])
        
        right = np.cross(f_norm, up_global)
        if np.linalg.norm(right) < 0.001: 
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
            
        up_cam = np.cross(right, f_norm)
        abertura = dist * 0.4 
        
        c1 = A + (right * abertura) + (up_cam * abertura)
        c2 = A - (right * abertura) + (up_cam * abertura)
        c3 = A - (right * abertura) - (up_cam * abertura)
        c4 = A + (right * abertura) - (up_cam * abertura)
        
        cantos = [c1, c2, c3, c4, c1] 
        
        for canto in cantos[:-1]:
            eixo_3d.plot([E[0], canto[0]], [E[1], canto[1]], [E[2], canto[2]], color='gray', linestyle='--', alpha=0.5)
            
        eixo_3d.plot([c[0] for c in cantos], [c[1] for c in cantos], [c[2] for c in cantos], color='gray', linewidth=2, label='Área de Visão')

    # 4. Mover a câmera real do Matplotlib
    dx = ex - ax_at
    dy = ez - az_at
    dz = ey - ay_at
    
    r = np.sqrt(dx**2 + dy**2)
    #elev = np.degrees(np.arctan2(dz, r))
    #azim = np.degrees(np.arctan2(dy, dx)+45)
    #eixo_3d.view_init(elev=elev, azim=azim)

    # 5. Rótulos e Atualização
    eixo_3d.set_xlabel('Eixo X')
    eixo_3d.set_ylabel('Eixo Z')
    eixo_3d.set_zlabel('Eixo Y')
    #eixo_3d.legend()


    # ==========================================
    # MANTER A ESCALA PROPORCIONAL (EVITAR DEFORMAÇÃO)
    # ==========================================
    # 1. Força a caixa do gráfico a desenhar um cubo perfeito
    eixo_3d.set_box_aspect([1, 1, 1])
    
    # 2. Pega os limites atuais que o Matplotlib calculou para incluir todos os pontos
    x_lim = eixo_3d.get_xlim3d()
    y_lim = eixo_3d.get_ylim3d() # Lembrando que fisicamente esse é o profundidade
    z_lim = eixo_3d.get_zlim3d() # Fisicamente a altura no matplotlib
    
    # 3. Descobre qual eixo teve que se esticar mais e cria um raio uniforme
    raio_maximo = max([
        abs(x_lim[1] - x_lim[0]),
        abs(y_lim[1] - y_lim[0]),
        abs(z_lim[1] - z_lim[0])
    ]) / 2.0
    
    # 4. Encontra o ponto central exato de cada eixo
    x_centro = np.mean(x_lim)
    y_centro = np.mean(y_lim)
    z_centro = np.mean(z_lim)
    
    # 5. Aplica a mesma distância para todos os lados a partir do centro
    eixo_3d.set_xlim3d([x_centro - raio_maximo, x_centro + raio_maximo])
    eixo_3d.set_ylim3d([y_centro - raio_maximo, y_centro + raio_maximo])
    eixo_3d.set_zlim3d([z_centro - raio_maximo, z_centro + raio_maximo])

    canvas.draw()


# ==========================================
#        CONFIGURAÇÃO DA INTERFACE (UI)
# ==========================================
janela = tk.Tk()
janela.title("Simulador de Câmera 3D Profissional")
janela.geometry("950x650")

sidebar = tk.Frame(janela, width=280, bg="#e0e0e0", padx=15, pady=15)
sidebar.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(sidebar, text="Controles da Câmera", font=("Arial", 12, "bold"), bg="#e0e0e0").pack(pady=(0, 15))

# Campos EYE (Câmera)
tk.Label(sidebar, text="Eye (Posição da Câmera)", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(anchor="w")
frame_eye = tk.Frame(sidebar, bg="#e0e0e0")
frame_eye.pack(fill=tk.X, pady=5)
tk.Label(frame_eye, text="X:", bg="#e0e0e0").grid(row=0, column=0)
ent_eye_x = tk.Entry(frame_eye, width=5); ent_eye_x.insert(0, "0"); ent_eye_x.grid(row=0, column=1, padx=2)
tk.Label(frame_eye, text="Y:", bg="#e0e0e0").grid(row=0, column=2)
ent_eye_y = tk.Entry(frame_eye, width=5); ent_eye_y.insert(0, "0"); ent_eye_y.grid(row=0, column=3, padx=2)
tk.Label(frame_eye, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_eye_z = tk.Entry(frame_eye, width=5); ent_eye_z.insert(0, "20"); ent_eye_z.grid(row=0, column=5, padx=2)

# Campos AT (Alvo)
tk.Label(sidebar, text="At (Foco do Olhar)", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(anchor="w", pady=(15, 0))
frame_at = tk.Frame(sidebar, bg="#e0e0e0")
frame_at.pack(fill=tk.X, pady=5)
tk.Label(frame_at, text="X:", bg="#e0e0e0").grid(row=0, column=0)
ent_at_x = tk.Entry(frame_at, width=5); ent_at_x.insert(0, "0"); ent_at_x.grid(row=0, column=1, padx=2)
tk.Label(frame_at, text="Y:", bg="#e0e0e0").grid(row=0, column=2)
ent_at_y = tk.Entry(frame_at, width=5); ent_at_y.insert(0, "0"); ent_at_y.grid(row=0, column=3, padx=2)
tk.Label(frame_at, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_at_z = tk.Entry(frame_at, width=5); ent_at_z.insert(0, "-60"); ent_at_z.grid(row=0, column=5, padx=2)

# Campos LUZ (Posição da Fonte de Luz)
tk.Label(sidebar, text="Luz (Posição da Luz)", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(anchor="w", pady=(15, 0))
frame_luz = tk.Frame(sidebar, bg="#e0e0e0")
frame_luz.pack(fill=tk.X, pady=5)
tk.Label(frame_luz, text="X:", bg="#e0e0e0").grid(row=0, column=0)
ent_luz_x = tk.Entry(frame_luz, width=5); ent_luz_x.insert(0, "0"); ent_luz_x.grid(row=0, column=1, padx=2)
tk.Label(frame_luz, text="Y:", bg="#e0e0e0").grid(row=0, column=2)
ent_luz_y = tk.Entry(frame_luz, width=5); ent_luz_y.insert(0, "40"); ent_luz_y.grid(row=0, column=3, padx=2)
tk.Label(frame_luz, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_luz_z = tk.Entry(frame_luz, width=5); ent_luz_z.insert(0, "20"); ent_luz_z.grid(row=0, column=5, padx=2)



# --- NOVIDADE: Checkbox para Manter Direção ---
var_manter_direcao = tk.BooleanVar(value=True) # Começa ativado por padrão
chk_direcao = tk.Checkbutton(sidebar, text="Recalcular At (Acompanhar Movimento)", 
                             variable=var_manter_direcao, bg="#e0e0e0", font=("Arial", 9))
chk_direcao.pack(anchor="w", pady=(10, 0))

var_mostrar_scc = tk.BooleanVar(value=False)

chk_scc = tk.Checkbutton(
    sidebar,
    text="Mostrar em SCC",
    variable=var_mostrar_scc,
    bg="#e0e0e0"
)

chk_scc.pack(anchor="w")
# Botão de Ação
btn_atualizar = tk.Button(sidebar, text="Atualizar Visualização", command=atualizar_grafico, bg="#0052cc", fg="white", font=("Arial", 10, "bold"))
btn_atualizar.pack(fill=tk.X, pady=25)

# Configuração do botão na interface
btn_rasterizar = tk.Button(sidebar, text="Rasterizar (3 Resoluções)", command=disparar_botao_rasterizar, bg="#cc6600", fg="white", font=("Arial", 10, "bold"))
btn_rasterizar.pack(fill=tk.X, pady=10)

# --- Área do Gráfico ---
frame_direito = tk.Frame(janela)
frame_direito.pack(
    side=tk.RIGHT,
    fill=tk.BOTH,
    expand=True
)

# 50% para o 3D
frame_direito.rowconfigure(0, weight=1)

# 50% para o 2D
frame_direito.rowconfigure(1, weight=1)

frame_direito.columnconfigure(0, weight=1)

frame_3d = tk.Frame(frame_direito)
frame_3d.grid(
    row=0,
    column=0,
    sticky="nsew"
)

frame_2d = tk.Frame(frame_direito)
frame_2d.grid(
    row=1,
    column=0,
    sticky="nsew"
)
frame_3d.config(bg="red")
frame_2d.config(bg="blue")

fig = plt.figure(figsize=(4, 3))
eixo_3d = fig.add_subplot(111, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=frame_3d)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas_2d = tk.Canvas(
    frame_2d,
    bg="white",
    height=250
)

canvas_2d.pack(
    fill=tk.BOTH,
    expand=True
)



atualizar_grafico()
janela.mainloop()