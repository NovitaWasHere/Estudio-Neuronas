class MiClase:
    def __init__(self, valor):
        self.valor = valor  # Accediendo y asignando el atributo "valor" de la instancia actual

    def metodo1(self):
        print("Este es el método 1")
        print(self.valor)  # Accediendo al atributo "valor" de la instancia actual

    def metodo2(self, otro_valor):
        self.valor = otro_valor  # Asignando un nuevo valor al atributo "valor" de la instancia actual

# Creando una instancia de la clase
objeto = MiClase("Hola")

# Llamando a los métodos de la instancia
objeto.metodo1()  # Salida: Este es el método 1 \n Hola
objeto.metodo2("Adiós")

objeto.metodo1()  # Salida: Este es el método 1 \n Adiós
