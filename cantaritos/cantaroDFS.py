from collections import defaultdict

metaAlcanz = False
camino = []

# Constructor
class Grafo:
    def __init__(self):
        # Diccionario por defecto para almacenar el grafo
        self.grafo = defaultdict(list)


    # Función para agregar una arista al grafo
    def agregarArista(self, u, v):
        self.grafo[u].append(v)


    # Una función utilizada por DFS
    def DFSUtil(self, v, visitados, meta):
        # Marcar el nodo actual como visitado
        # e imprimirlo
        visitados.add(v)
        #Guarda el nodo actual en el camino
        camino.append(v)
        #print(v, end=' ')
        global metaAlcanz
        #Si el nodo actual es la meta entonces activa la bandera
        if(v==meta):
            metaAlcanz=True
        # Recorrer todos los vértices
        # adyacentes a este vértice
        for vecino in self.grafo[v]:
            if metaAlcanz==False:
                if vecino not in visitados:
                    self.DFSUtil(vecino, visitados, meta)
                    if(metaAlcanz==False):
                        #Si al terminar de recorrer toda la rama aun no se alcanzo la meta entonces saca los nodos del camino
                        camino.pop()


    # La función para hacer el recorrido DFS. Utiliza
    # DFSUtil() de manera recursiva
    def DFS(self, v, meta):
        # Crear un conjunto para almacenar los vértices visitados
        visitados = set()

        # Llamar a la función auxiliar recursiva
        # para imprimir el recorrido DFS
        self.DFSUtil(v, visitados, meta)
        print("")
        for x in camino:
            print(x, end=' ')

if __name__ == "__main__":
    #A=0 D=1 H=2 B=3 J=4
    #K=5 L=6 F=7 C=8 E=9 Z=10 W=11 G=12
    g = Grafo()
    g.agregarArista('A','D')
    g.agregarArista('A','F')
    g.agregarArista('A','G')
    
    g.agregarArista('D','H')
    g.agregarArista('D','J')
    
    g.agregarArista('H','B')
    
    g.agregarArista('J','K')
    
    g.agregarArista('K','L')
    
    g.agregarArista('F','C')
    g.agregarArista('F','E')
    
    g.agregarArista('E','Z')
    g.agregarArista('E','W')

print("A continuación se muestra el recorrido DFS (comenzando desde el vértice A)")

# Llamada a la función
g.DFS('A','L')
