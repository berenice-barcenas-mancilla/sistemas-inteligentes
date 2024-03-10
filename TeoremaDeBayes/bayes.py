import tkinter as tk

def calcular_probabilidad():
    try:
        # Obtener los valores ingresados por el usuario
        prob_incidente = float(prob_incidente_entry.get())
        prob_alarma_dado_incidente = float(prob_alarma_dado_incidente_entry.get())
        prob_alarma_dado_no_incidente = float(prob_alarma_dado_no_incidente_entry.get())
        
        # Calcular la probabilidad de que la alarma suene
        prob_alarma = prob_alarma_dado_incidente * prob_incidente + prob_alarma_dado_no_incidente * (1 - prob_incidente)
        
        # Calcular la probabilidad de que no haya habido ningún incidente dado que la alarma ha sonado
        prob_no_incidente_dado_alarma = (prob_alarma_dado_no_incidente * (1 - prob_incidente)) / prob_alarma
        
        # Actualizar la etiqueta con el resultado
        resultado_label.config(text=f"La probabilidad de que no haya habido ningún incidente dado que la alarma ha sonado es aproximadamente: {prob_no_incidente_dado_alarma:.4f}")
    except ValueError:
        resultado_label.config(text="Por favor, ingrese valores válidos.")

# Crear la ventana
root = tk.Tk()
root.title("Calculadora de Probabilidad - Teorema de Bayes")

# Crear los widgets
prob_incidente_label = tk.Label(root, text="Probabilidad de que haya un incidente:")
prob_incidente_entry = tk.Entry(root)
prob_alarma_dado_incidente_label = tk.Label(root, text="Probabilidad de que la alarma suene dado que ha habido un incidente:")
prob_alarma_dado_incidente_entry = tk.Entry(root)
prob_alarma_dado_no_incidente_label = tk.Label(root, text="Probabilidad de que la alarma suene dado que no ha habido ningún incidente:")
prob_alarma_dado_no_incidente_entry = tk.Entry(root)
calcular_button = tk.Button(root, text="Calcular Probabilidad", command=calcular_probabilidad)
resultado_label = tk.Label(root, text="Resultado aquí")

# Ubicar los widgets en la ventana
prob_incidente_label.grid(row=0, column=0, sticky="e")
prob_incidente_entry.grid(row=0, column=1)
prob_alarma_dado_incidente_label.grid(row=1, column=0, sticky="e")
prob_alarma_dado_incidente_entry.grid(row=1, column=1)
prob_alarma_dado_no_incidente_label.grid(row=2, column=0, sticky="e")
prob_alarma_dado_no_incidente_entry.grid(row=2, column=1)
calcular_button.grid(row=3, columnspan=2)
resultado_label.grid(row=4, columnspan=2)

# Ejecutar el bucle principal
root.mainloop()
