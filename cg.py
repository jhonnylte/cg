import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

      self.atualizar_normais()
      return Objeto3D(vertices=Pn, faces=self.faces, normais=self.normais)

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
      Pn = np.dot(ViewMatrix, adicionar_CH(self.vertices).T).T
      Pn = Pn[:, :3]
      objetoScc = Objeto3D(
          vertices = Pn,
          normais = self.normais,
          faces = self.faces
      )
      return objetoScc

    def atualizar_normais(self):
      self.normais = calcular_normais_faces(self.vertices, self.faces)


""" CRIAÇÃO DO PEÃO COM MARCHING CUBES """
N = 8

x = np.linspace(-4, 4, N)
y = np.linspace(0, 12, N)
z = np.linspace(-4, 4, N)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing='ij'
)

cilindro = np.sqrt(X**2 + Z**2) - 3
cilindro[Y > 2] = 1

raio_tronco = 2 - (Y - 4)/4
tronco = np.sqrt(X**2 + Z**2) - raio_tronco
tronco[(Y < 2) | (Y > 8)] = 1

esfera = np.sqrt(X**2 + (Y - 9.5)**2 + Z**2) - 2

volume = np.minimum(cilindro, tronco)
volume = np.minimum(volume, esfera)

peao_vertices, peao_faces, peao_normais, _ = marching_cubes(
    volume,
    level=0
)

peao_vertices[:,0] = np.interp(
    peao_vertices[:,0],
    [0, N-1],
    [x.min(), x.max()]
)

peao_vertices[:,1] = np.interp(
    peao_vertices[:,1],
    [0, N-1],
    [y.min(), y.max()]
)

peao_vertices[:,2] = np.interp(
    peao_vertices[:,2],
    [0, N-1],
    [z.min(), z.max()]
)

peao = Objeto3D(
    vertices=peao_vertices,
    faces=peao_faces,
    normais=peao_normais
)
""""====================PEÃO CRIADO COM MARCHING CUBES===================="""
"""======================================================================="""






""" Variável global para guardar a posição anterior da câmera """
ultimo_eye = [10.0, 20.0, 10.0]

def atualizar_grafico():
    global ultimo_eye
    
    try:
        ex, ey, ez = float(ent_eye_x.get()), float(ent_eye_y.get()), float(ent_eye_z.get())
        ax_at, ay_at, az_at = float(ent_at_x.get()), float(ent_at_y.get()), float(ent_at_z.get())
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
    ultimo_eye = [ex, ey, ez]


    # ==========================================
    # DESENHO DO GRÁFICO (CANVAS)
    # ==========================================
    eixo_3d.cla()

    # 1. Trajetória (Espiral)
    #t = np.linspace(0, 20, 200)
    #eixo_3d.plot(np.sin(t), np.cos(t), t, color='teal', linewidth=2.5, alpha=0.5, label='Trajetória')

    # 2. Desenhar Alvo e Câmera
    eixo_3d.scatter(ax_at, az_at, ay_at, color='red', s=150, marker='.', label='')
    eixo_3d.scatter(ex, ez, ey, color='black', s=100, marker='.', label='')
    eixo_3d.scatter(0, 0, 0, color='green', s=200, marker='.', label='Origem (0,0,0)')

    """ Desenhar o Peão usando os vértices gerados pelo marching cubes """
    Vpeao = peao.vertices

    triangulosPeao = []

    for face in peao.faces:

        triangulo = peao.vertices[face]

        triangulo = triangulo[:, [0,2,1]]

        triangulosPeao.append(triangulo)

    malhaPeao = Poly3DCollection(
        triangulosPeao,
        edgecolor='blue',
        linewidth=0.2,
        alpha=0.8
    )

    eixo_3d.add_collection3d(malhaPeao)

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
    eixo_3d.legend()


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
# LÓGICA DE SCRUBBING (ARRASTAR PARA MUDAR)
# ==========================================
def iniciar_scrub(event):
    """Armazena a posição inicial do mouse no momento do clique e o valor atual do campo."""
    event.widget.start_x = event.x
    try:
        event.widget.start_val = float(event.widget.get())
    except ValueError:
        event.widget.start_val = 0.0

def executar_scrub(event):
    """Calcula a diferença do mouse, altera o valor e atualiza o gráfico em tempo real."""
    # Calcula o deslocamento do mouse (multiplique por 0.2 para ajustar a sensibilidade)
    delta = (event.x - event.widget.start_x) * 0.2 
    novo_valor = event.widget.start_val + delta
    
    # Atualiza o texto dentro da caixinha
    event.widget.delete(0, tk.END)
    event.widget.insert(0, str(round(novo_valor, 2)))
    
    # Chama a função principal para redesenhar a tela
    atualizar_grafico()

def atualizar_digitacao(event):
    """Garante que o gráfico também atualize se você apenas digitar um número."""
    # Ignora teclas de controle (como setas, shift, etc) para não atualizar à toa
    if event.char != '':
        atualizar_grafico()

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
ent_eye_x = tk.Entry(frame_eye, width=5); ent_eye_x.insert(0, "15"); ent_eye_x.grid(row=0, column=1, padx=2)
tk.Label(frame_eye, text="Y:", bg="#e0e0e0").grid(row=0, column=2)
ent_eye_y = tk.Entry(frame_eye, width=5); ent_eye_y.insert(0, "20"); ent_eye_y.grid(row=0, column=3, padx=2)
tk.Label(frame_eye, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_eye_z = tk.Entry(frame_eye, width=5); ent_eye_z.insert(0, "15"); ent_eye_z.grid(row=0, column=5, padx=2)

# Campos AT (Alvo)
tk.Label(sidebar, text="At (Foco do Olhar)", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(anchor="w", pady=(15, 0))
frame_at = tk.Frame(sidebar, bg="#e0e0e0")
frame_at.pack(fill=tk.X, pady=5)
tk.Label(frame_at, text="X:", bg="#e0e0e0").grid(row=0, column=0)
ent_at_x = tk.Entry(frame_at, width=5); ent_at_x.insert(0, "0"); ent_at_x.grid(row=0, column=1, padx=2)
tk.Label(frame_at, text="Y:", bg="#e0e0e0").grid(row=0, column=2)
ent_at_y = tk.Entry(frame_at, width=5); ent_at_y.insert(0, "10"); ent_at_y.grid(row=0, column=3, padx=2)
tk.Label(frame_at, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_at_z = tk.Entry(frame_at, width=5); ent_at_z.insert(0, "0"); ent_at_z.grid(row=0, column=5, padx=2)


# ... (seu código atual) ...
tk.Label(frame_eye, text="Z:", bg="#e0e0e0").grid(row=0, column=4)
ent_eye_z = tk.Entry(frame_eye, width=5); ent_eye_z.insert(0, "15"); ent_eye_z.grid(row=0, column=5, padx=2)

# --- NOVIDADE: Adicionando os eventos de Scrubbing e Digitação nos campos EYE ---
for campo in (ent_eye_x, ent_eye_y, ent_eye_z):
    # Quando o usuário clica e segura
    campo.bind("<ButtonPress-1>", iniciar_scrub)
    # Quando o usuário arrasta o mouse pressionado para a esquerda/direita
    campo.bind("<B1-Motion>", executar_scrub)
    # Quando o usuário solta uma tecla após digitar um número manualmente
    campo.bind("<KeyRelease>", atualizar_digitacao)


# --- NOVIDADE: Checkbox para Manter Direção ---
var_manter_direcao = tk.BooleanVar(value=True) # Começa ativado por padrão
chk_direcao = tk.Checkbutton(sidebar, text="Recalcular At (Acompanhar Movimento)", 
                             variable=var_manter_direcao, bg="#e0e0e0", font=("Arial", 9))
chk_direcao.pack(anchor="w", pady=(10, 0))

# Botão de Ação
btn_atualizar = tk.Button(sidebar, text="Atualizar Visualização", command=atualizar_grafico, bg="#0052cc", fg="white", font=("Arial", 10, "bold"))
btn_atualizar.pack(fill=tk.X, pady=25)

# --- Área do Gráfico ---
frame_grafico = tk.Frame(janela)
frame_grafico.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig = plt.figure(figsize=(6, 5))
eixo_3d = fig.add_subplot(111, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

atualizar_grafico()
janela.mainloop()