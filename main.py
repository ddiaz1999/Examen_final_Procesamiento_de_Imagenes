from bandera import*

if __name__ == '__main__':
    path =  input('ingrese path: ') #r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'
    image_name =  input('ingrese imagen: ') #'flag5.png'
    path_file = os.path.join(path, image_name)

    image = bandera(path_file)
    image.colores()
    image.porcentaje()
    image.orientacion()


