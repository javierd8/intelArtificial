from collections import defaultdict

metaAlcanz = False
camino = []

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
                    if metaAlcanz==False:
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
    #Cantaritos (Arbol de 2 ramas)
    g = Grafo()
    #Lvl1
    g.agregarArista('0,0','4,0')
    g.agregarArista('0,0','0,3')
    #Lvl2
    g.agregarArista('4,0','1,3') 
    g.agregarArista('0,3','3,0')
    #Lvl3
    g.agregarArista('1,3','4,3')
    g.agregarArista('3,0','3,3')
    #Lvl4
    g.agregarArista('4,3','0,3')
    g.agregarArista('3,3','4,2')
    #Lvl5
    g.agregarArista('0,3','3,0') 
    g.agregarArista('4,2','4,3')
    #Lvl6
    g.agregarArista('3,0','3,3') 
    g.agregarArista('4,3','4,0')
    #Lvl7
    g.agregarArista('3,3','4,2') 
    g.agregarArista('4,0','1,3')
    #Lvl8
    g.agregarArista('4,2','0,2') 
    g.agregarArista('1,3','1,0')
    #Lvl9
    g.agregarArista('0,2','2,0')
    g.agregarArista('1,0','0,1')
    #Lvl10
    g.agregarArista('2,0','2,3')
    g.agregarArista('0,1','4,1')
    #Lvl11
    g.agregarArista('2,3','4,1')
    g.agregarArista('4,1','2,3')
    #Lvl12
    g.agregarArista('4,1','0,1')
    g.agregarArista('2,3','2,0')
    #Lvl13
    g.agregarArista('0,1','1,0')
    g.agregarArista('2,0','0,2')
    #Lvl14
    g.agregarArista('1,0','1,3')


print("A continuación se muestra el recorrido DFS (comenzando desde el vértice 0,0)")

# Llamada a la función (Para que una busqueda sea valida la meta debe tener un min(0) o max(4 o 3 dependiendo) en alguno de los cantaros)
g.DFS('0,0','4,3')
if not metaAlcanz:
    print("La meta no es posible de alcanzar(No es valida)")
