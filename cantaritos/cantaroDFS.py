from collections import defaultdict


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
        print(v, end=' ')
        global metaAlcanz
        if(v==meta):
            metaAlcanz=True
        # Recorrer todos los vértices
        # adyacentes a este vértice
        for vecino in self.grafo[v]:
            if metaAlcanz==False:
                if vecino not in visitados:
                    self.DFSUtil(vecino, visitados, meta)
                else:
                    visitados.remove(v)


    # La función para hacer el recorrido DFS. Utiliza
    # DFSUtil() de manera recursiva
    def DFS(self, v, meta):
        # Crear un conjunto para almacenar los vértices visitados
        visitados = set()

        # Llamar a la función auxiliar recursiva
        # para imprimir el recorrido DFS
        self.DFSUtil(v, visitados, meta)
        print("")
        for x in visitados:
            print(x, end=' ')

metaAlcanz = False
if __name__ == "__main__":
    #A=0 D=1 H=2 B=3 J=4
    #K=5 L=6 F=7 C=8 E=9 Z=10 W=11 G=12
    g = Grafo()
    g.agregarArista(0,1)
    g.agregarArista(0,7)
    g.agregarArista(0,12)
    g.agregarArista(1,2)
    g.agregarArista(1,4)
    g.agregarArista(2,3)
    g.agregarArista(2,3)
    g.agregarArista(4,5)
    g.agregarArista(5,6)
    g.agregarArista(7,8)
    g.agregarArista(7,9)
    g.agregarArista(9,10)
    g.agregarArista(9,11)

print("A continuación se muestra el recorrido DFS (comenzando desde el vértice 0)")

# Llamada a la función
g.DFS(0,6)
