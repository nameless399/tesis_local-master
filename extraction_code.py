import os

# ================= CONFIGURACIÓN =================
# Nombre del archivo donde se guardará todo
ARCHIVO_SALIDA = "codigo_para_ia.txt"

# Carpetas que la IA NO necesita ver (para ahorrar espacio y no confundirla)
CARPETAS_IGNORADAS = {
    '.git', '.vscode', '.idea', '__pycache__', 
    'node_modules', 'venv', 'env', 'dist', 'build'
}

# Extensiones de archivos binarios o irrelevantes que se van a saltar
EXTENSIONES_IGNORADAS = {
    '.exe', '.png', '.jpg', '.jpeg', '.gif', '.pdf', 
    '.zip', '.tar', '.gz', '.mp4', '.mp3', '.pyc', 
    '.sqlite3', '.ico', '.svg', '.lock', '.sql', '.npz','.mp4','.avi','.pkl','.keras'
}
# =================================================

def extraer_codigo():
    # Obtener el nombre de este mismo script para no incluirlo en el resultado
    script_actual = os.path.basename(__file__)
    ruta_base = os.getcwd()

    archivos_procesados = 0

    print("Extrayendo archivos, por favor espera... :D")

    with open(ARCHIVO_SALIDA, 'w', encoding='utf-8') as archivo_out:
        for root, dirs, files in os.walk(ruta_base):
            
            # Modificamos la lista 'dirs' para que os.walk no entre en las carpetas ignoradas
            dirs[:] = [d for d in dirs if d not in CARPETAS_IGNORADAS]

            for file in files:
                # Ignorar este script y el archivo de salida
                if file == script_actual or file == ARCHIVO_SALIDA:
                    continue

                # Ignorar extensiones no deseadas
                _, ext = os.path.splitext(file)
                if ext.lower() in EXTENSIONES_IGNORADAS:
                    continue

                ruta_completa = os.path.join(root, file)
                # Obtenemos la ruta relativa para que sea más fácil de leer para la IA
                ruta_relativa = os.path.relpath(ruta_completa, ruta_base)

                try:
                    # Intentamos leer el archivo como texto
                    with open(ruta_completa, 'r', encoding='utf-8') as archivo_in:
                        contenido = archivo_in.read()

                    # ===== FORMATO PARA LA IA =====
                    # Esto le ayuda a la IA a saber dónde empieza y termina cada archivo
                    archivo_out.write(f"{'='*80}\n")
                    archivo_out.write(f"Archivo: {ruta_relativa}\n")
                    archivo_out.write(f"{'='*80}\n\n")
                    archivo_out.write(contenido)
                    archivo_out.write("\n\n")
                    
                    archivos_procesados += 1

                except UnicodeDecodeError:
                    # Si da este error, probablemente es un archivo binario (ej. imagen) que no filtramos. Lo saltamos.
                    pass
                except Exception as e:
                    # Si hay algún otro error (ej. permisos), lo anotamos pero seguimos
                    archivo_out.write(f"// Error al leer {ruta_relativa}: {e}\n\n")

    print(f"¡Listo! :D Se extrajeron {archivos_procesados} archivos.")
    print(f"Busca el archivo '{ARCHIVO_SALIDA}' en esta misma carpeta.")

if __name__ == "__main__":
    extraer_codigo()

