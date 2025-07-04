#  TFG - Mejora del modelo MMGIN para predicción de toxicidad molecular

Este repositorio contiene el código y recursos asociados al Trabajo de Fin de Grado de **Pablo López Outes**. El objetivo principal es el **análisis y mejora del modelo MMGIN**, originalmente propuesto para predicción multitarea de toxicidad molecular, adaptándolo al contexto de clasificación binaria y explorando variantes arquitectónicas modernas.

---

##  Objetivos del proyecto

El propósito general de este trabajo es evaluar y rediseñar el modelo MMGIN con técnicas del estado del arte para mejorar su capacidad predictiva en tareas de clasificación binaria (tóxica / no tóxica).

---

##  Configuración del entorno

Para garantizar la correcta ejecución del código, se recomienda utilizar un entorno virtual con **Miniconda**. A continuación se describen los pasos para sistemas Linux:

### 1. Instalar Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
sh Miniconda3-py39_4.11.0-Linux-x86_64.sh
conda init


## Crear y activar el entorno Conda
Una vez instalado Miniconda y reiniciada la terminal, puedes crear el entorno ejecutando:


conda env create -f mmgin.yml
conda activate mmgin


---


## Ejecución
Con el entorno activado, puedes ejecutar cualquier script del repositorio. Por ejemplo:

python binary_classificationCrossValAutomatico.py



