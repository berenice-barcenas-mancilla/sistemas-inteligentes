import tkinter as tk

def calcular_probabilidad():
    try:
        #obtiene los valores ingresados por el usuario
        probabilidades = []
        for entry in entries:
            probabilidad = float(entry.get())
            probabilidades.append(probabilidad)
        
        #calcula la probabilidad de que la alarma suene
        prob_alarma = sum(probabilidades[1::2]) * probabilidades[0] + sum(probabilidades[2::2]) * (1 - probabilidades[0])
        
        #calcula la probabilidad de que no haya habido ningún incidente dado que la alarma ha sonado
        prob_no_incidente_dado_alarma = (sum(probabilidades[2::2]) * (1 - probabilidades[0])) / prob_alarma
        
        #muestra el resultado en una nueva ventana
        mostrar_resultado(prob_no_incidente_dado_alarma)
    except ValueError:
        resultado_label.config(text="Por favor, ingrese valores válidos.")

def mostrar_resultado(probabilidad_decimal):
    #crea una nueva ventana para mostrar el resultado
    resultado_window = tk.Toplevel(root)
    resultado_window.title("Resultado")
    
    #convierte el resultado en porcentaje
    probabilidad_porcentaje = probabilidad_decimal * 100
    
    #muestra el resultado en forma decimal y en forma de porcentaje
    resultado_decimal_label = tk.Label(resultado_window, text=f"La probabilidad de que no haya habido ningún incidente dado
                                       que la alarma ha sonado es aproximadamente: {probabilidad_decimal:.4f}")
    resultado_porcentaje_label = tk.Label(resultado_window, text=f"En porcentaje: {probabilidad_porcentaje:.2f}%")
    
    resultado_decimal_label.pack()
    resultado_porcentaje_label.pack()

def agregar_campos():
    num_campos = int(num_eventos_entry.get())
    #Limpia los widgets anteriores
    for entry in entries:
        entry.destroy()
    entries.clear()
    #crear nuevos widgets para ingresar las probabilidades
    for i in range(num_campos * 2 + 1):
        if i % 2 == 0:
            label_text = f"Probabilidad del evento {i // 2 + 1}:"
        else:
            label_text = f"Probabilidad de la alarma dado el evento {i // 2 + 1}:"
        label = tk.Label(root, text=label_text)
        label.grid(row=i + 5, column=0, sticky="e")
        entry = tk.Entry(root)
        entry.grid(row=i + 5, column=1)
        entries.append(entry)

#crea la ventana principal
root = tk.Tk()
root.title("Calculadora de Probabilidad - Teorema de Bayes")

#crea los widgets
num_eventos_label = tk.Label(root, text="Número de eventos:")
num_eventos_entry = tk.Entry(root)
agregar_campos_button = tk.Button(root, text="Agregar campos", command=agregar_campos)
calcular_button = tk.Button(root, text="Calcular Probabilidad", command=calcular_probabilidad)
resultado_label = tk.Label(root, text="Resultado aquí")

#ubica los widgets en la ventana principal
num_eventos_label.grid(row=0, column=0, sticky="e")
num_eventos_entry.grid(row=0, column=1)
agregar_campos_button.grid(row=1, columnspan=2)
calcular_button.grid(row=2, columnspan=2)
resultado_label.grid(row=3, columnspan=2)

entries = []

#ejecuta el bucle principal
root.mainloop()
