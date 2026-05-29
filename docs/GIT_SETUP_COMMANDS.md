# Comandos Git Exactos para Cada Integrante

## IMPORTANTE: Leer todo esto ANTES de hacer cualquier comando

Este documento tiene los comandos exactos que cada integrante debe ejecutar para:
1. Clonar el repositorio
2. Crear su rama de trabajo
3. Hacer commits
4. Hacer push
5. Crear Pull Request

---

## SETUP INICIAL (Ejecutar una sola vez)

### Configurar Git (si es primera vez en la máquina)

```bash
git config --global user.name "Tu Nombre Completo"
git config --global user.email "tu.email@ejemplo.com"
```

Verificar:
```bash
git config --global --list
```

---

## PASO 1: Clonar el repositorio

Ejecutar SOLO una vez:

```bash
cd C:\Users\ashar\Documents
git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
cd pi-bbva-mexico-ml
```

Verificar que está clonado:
```bash
git status
# Debe mostrar: On branch main
```

---

## PASO 2: Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Generar splits (ejecutar una sola vez):
```bash
python src/utils/splits.py --input data/raw/BasePI.xlsx --output data/processed/
```

---

## PASO 3: Crear rama personal

### Para P1 (Regresión Logística):
```bash
git checkout -b P1/woe-scorecard
```

### Para P2 (XGBoost):
```bash
git checkout -b P2/xgboost-tuning
```

### Para P3 (Random Forest):
```bash
git checkout -b P3/random-forest-features
```

### Para P4 (Red Neuronal):
```bash
git checkout -b P4/neural-network
```

Verificar:
```bash
git branch
# Debe mostrar tu rama con asterisco (*)
```

---

## PASO 4: Trabajar en notebooks

### Crear/editar tus notebooks

Por ejemplo, P1 edita:
- `notebooks/P1_Logistic_Regression/01_P1_EDA_LR.ipynb`
- `notebooks/P1_Logistic_Regression/02_P1_WOE_Engineering.ipynb`
- etc.

```bash
# Ver cambios
git status

# Ver cambios en detalle
git diff
```

---

## PASO 5: Hacer commits (frecuentemente)

Después de completar una sección de tu notebook:

```bash
# Ver qué has modificado
git status

# Agregar cambios
git add .

# Hacer commit con mensaje siguiendo convención
git commit -m "[P1] feat: implementar WOE para variables continuas"
```

### Ejemplos por integrante:

**P1**:
```bash
git commit -m "[P1] feat: crear función de binning automático"
git commit -m "[P1] feat: calcular Weight of Evidence por variable"
git commit -m "[P1] feat: generar scorecard con puntos"
```

**P2**:
```bash
git commit -m "[P2] feat: implementar baseline XGBoost"
git commit -m "[P2] feat: tuning hiperparámetros con Optuna"
git commit -m "[P2] feat: análisis de feature importance"
```

**P3**:
```bash
git commit -m "[P3] feat: análisis bivariado completo"
git commit -m "[P3] feat: entrenar modelo Random Forest"
git commit -m "[P3] feat: comparar importancia de features"
```

**P4**:
```bash
git commit -m "[P4] feat: análisis univariado completo"
git commit -m "[P4] feat: normalización de datos"
git commit -m "[P4] feat: entrenar red neuronal MLP"
```

Ver historial de commits:
```bash
git log --oneline
# Mostrar últimos 10 commits
git log --oneline -n 10
```

---

## PASO 6: Actualizar rama con cambios de main (si hay)

Antes de hacer push, asegúrate de que tu rama está actualizada:

```bash
# Traer cambios del repositorio remoto
git fetch origin

# Ver si hay cambios en main
git log origin/main --oneline -n 5

# Actualizar tu rama con los cambios de main
git rebase origin/main
```

Si hay conflictos, Git te lo dirá. Contactar a P4 para ayuda.

---

## PASO 7: Hacer Push (subir cambios a GitHub)

### Primer push de tu rama (crear en remoto)
```bash
git push -u origin P1/woe-scorecard    # P1
git push -u origin P2/xgboost-tuning   # P2
git push -u origin P3/random-forest-features  # P3
git push -u origin P4/neural-network   # P4
```

### Pushes posteriores (una vez creada la rama)
```bash
git push origin P1/woe-scorecard    # P1
git push origin P2/xgboost-tuning   # P2
git push origin P3/random-forest-features  # P3
git push origin P4/neural-network   # P4
```

Verificar en GitHub:
1. Ir a https://github.com/Asharetbe/pi-bbva-mexico-ml
2. Ver que tu rama aparece en "Branches"

---

## PASO 8: Crear Pull Request en GitHub

Después del push, en GitHub:

1. **Ir a** https://github.com/Asharetbe/pi-bbva-mexico-ml/pulls

2. **Click en "New Pull Request"**

3. **Seleccionar ramas**:
   - Base: `main` (lado izquierdo)
   - Compare: `P1/woe-scorecard` (lado derecho, tu rama)

4. **Click "Create Pull Request"**

5. **Llenar información**:
   ```
   Título: Implementar modelo Regresión Logística con WOE
   
   Descripción:
   ## Cambios principales
   - Implementar WOE para todas las variables
   - Generar scorecard
   - Validar en train, test, OOT
   
   ## Archivos modificados
   - notebooks/P1_Logistic_Regression/01_P1_EDA_LR.ipynb
   - notebooks/P1_Logistic_Regression/02_P1_WOE_Engineering.ipynb
   - notebooks/P1_Logistic_Regression/03_P1_Model_Training.ipynb
   
   ## Notas
   - Model evaluado en test: AUROC = 0.XXXX
   - Todas las métricas guardadas en results/
   ```

6. **Asignar revisor**: P4 (si es posible)

7. **Click "Create Pull Request"**

8. **Esperar a que P4 revise y apruebe**

---

## PASO 9: Después de que se mergea el PR

Una vez que P4 mergea tu PR a main:

```bash
# Actualizar tu rama main local
git checkout main
git pull origin main

# Ver que está todo integrado
git log --oneline -n 10
```

Si necesitas seguir trabajando en tu rama:
```bash
git checkout P1/woe-scorecard
git rebase origin/main
```

---

## Flujo completo en un día de trabajo

```bash
# Mañana: empezar sesión de trabajo
cd pi-bbva-mexico-ml
git fetch origin
git checkout P1/woe-scorecard
git rebase origin/main

# Durante el día: editar notebooks y hacer commits
# (editar archivos en tu IDE)
git add .
git commit -m "[P1] feat: ..."
# (más cambios y commits)
git add .
git commit -m "[P1] feat: ..."

# Antes de salir: hacer push
git push origin P1/woe-scorecard
```

---

## Comandos útiles para referencia

### Ver status
```bash
git status                          # Estado actual
git log --oneline -n 10             # Últimos 10 commits
git branch -a                       # Ver todas las ramas
git diff                            # Ver cambios sin stagear
git diff --cached                   # Ver cambios stageados
```

### Deshacer cambios (CUIDADO)
```bash
git reset --hard HEAD               # Deshacer cambios no commiteados
git revert <commit-hash>            # Crear un nuevo commit que deshace uno anterior
```

### Cambiar de rama
```bash
git checkout main                   # Ir a main
git checkout P1/woe-scorecard       # Ir a tu rama
```

### Ver cambios de alguien
```bash
git log --author="P1" --oneline     # Commits de P1
git log P1/woe-scorecard --oneline  # Commits en rama de P1
```

---

## Si algo sale mal

### "Tengo conflictos en rebase"
```bash
# Ver conflictos
git status

# Ver el archivo conflictivo
cat <filename>

# Decidir qué mantener (editar manualmente)
# Luego:
git add .
git rebase --continue

# Si quieres cancelar:
git rebase --abort
```

### "Hice un commit con mensaje incorrecto"
```bash
# Cambiar mensaje del último commit (antes de push)
git commit --amend -m "[P1] feat: mensaje correcto"
```

### "Necesito ver cambios de otros"
```bash
git fetch origin
git log P2/xgboost-tuning --oneline   # Ver commits de P2
```

---

## Checklist diario

Antes de hacer push al final del día:

```
□ Edité mis notebooks
□ Hice commits con mensajes [P#] type: descripción
□ Mi rama está actualizada: git rebase origin/main
□ No hay conflictos
□ Hice git push origin Px/branch-name
□ El PR se ve correcto en GitHub
```

---

## Fechas importantes

| Fecha | Hito |
|-------|------|
| **May 26** | Setup y splits ✓ |
| **May 27** | M1 y M2 entrenados |
| **May 28** | M3 y M4 entrenados |
| **May 29** | Informe integrado |
| **May 30** | ✅ Entrega final |

---

## ¿Preguntas o problemas?

Si algo no funciona o no entiende un comando:
- Contactar a P4 (gestor de ramas)
- Ver `docs/BRANCHING_STRATEGY.md` para más detalle
- Ver `docs/COMMIT_CONVENTION.md` para formato de commits

---

## Comandos rápidos copiar-pegar

### P1
```bash
git checkout -b P1/woe-scorecard
git push -u origin P1/woe-scorecard
```

### P2
```bash
git checkout -b P2/xgboost-tuning
git push -u origin P2/xgboost-tuning
```

### P3
```bash
git checkout -b P3/random-forest-features
git push -u origin P3/random-forest-features
```

### P4
```bash
git checkout -b P4/neural-network
git push -u origin P4/neural-network
```

Después, para cada commit:
```bash
git add .
git commit -m "[Px] type: descripción"
git push origin Px/branch-name
```
