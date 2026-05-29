# Estrategia de Ramas (Branching Strategy) - PI BBVA México

## Modelo: Feature Branch + Pull Request

```
main (rama principal - código listo para entrega)
├── P1/woe-scorecard (rama de desarrollo P1)
├── P2/xgboost-tuning (rama de desarrollo P2)
├── P3/random-forest-features (rama de desarrollo P3)
└── P4/neural-network (rama de desarrollo P4)
```

## Ramas principales

### `main`
- **Propósito**: Código listo para producción/entrega
- **Quién puede mergear**: P4 (integrador) y solo vía Pull Request aprobado
- **Protecciones**: 
  - Requiere PR aprobado
  - Todos los tests deben pasar (si existen)
  - Historia lineal
- **Frecuencia de actualización**: Solo en hitos finales (May 30)

### Ramas de trabajo individual

#### `P1/woe-scorecard`
- Responsable: P1
- Contenido: Regresión Logística, WOE, Scorecard
- Secciones: §2.1 Problemática, §3 Conclusiones
- **Mantener actualizada** con cambios de main

#### `P2/xgboost-tuning`
- Responsable: P2
- Contenido: XGBoost, Gradient Boosting, Tuning
- Secciones: §1.1 BBVA México, §2.4 Comparación
- **Mantener actualizada** con cambios de main

#### `P3/random-forest-features`
- Responsable: P3
- Contenido: Random Forest, Análisis bivariado
- Secciones: §2.2.2 Análisis bivariado
- **Mantener actualizada** con cambios de main

#### `P4/neural-network`
- Responsable: P4
- Contenido: Red Neuronal MLP, Integración documento
- Secciones: §2.2.1 Análisis univariado, Compilación final
- **Mantener actualizada** con cambios de main

---

## Flujo de trabajo para cada integrante

### Paso 1: Clonar repositorio (una sola vez)
```bash
git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
cd pi-bbva-mexico-ml
```

### Paso 2: Actualizar main localmente
```bash
git fetch origin
git checkout main
git pull origin main
```

### Paso 3: Crear rama personal a partir de main
```bash
# P1
git checkout -b P1/woe-scorecard

# P2
git checkout -b P2/xgboost-tuning

# P3
git checkout -b P3/random-forest-features

# P4
git checkout -b P4/neural-network
```

### Paso 4: Trabajar localmente
```bash
# Hacer cambios en notebooks, scripts, etc.
# Crear commits frecuentes y con buena descripción
git add .
git commit -m "[P1] feat: implementar WOE"

# Más cambios...
git add .
git commit -m "[P1] feat: generar scorecard"
```

### Paso 5: Mantener rama actualizada con main
```bash
# Traer cambios más recientes de main
git fetch origin
git rebase origin/main

# Si hay conflictos, resolverlos manualmente
# Luego continuar
git rebase --continue
```

### Paso 6: Hacer Push a la rama remota
```bash
# Primer push (crear rama en remoto)
git push -u origin P1/woe-scorecard

# Pushes posteriores
git push origin P1/woe-scorecard
```

### Paso 7: Crear Pull Request en GitHub
1. Ir a https://github.com/Asharetbe/pi-bbva-mexico-ml/pulls
2. Click en "New Pull Request"
3. **Base**: `main` ← **Compare**: `P1/woe-scorecard`
4. Llenar descripción con:
   - Resumen de cambios
   - Archivos modificados
   - Cualquier nota importante
5. Asignar revisor: P4 (integrador)
6. Click "Create Pull Request"

### Paso 8: Revisión y Merge
- P4 revisa el PR
- Commenta si hay cambios necesarios
- Si todo está OK, aprueba y mergea a main
- La rama P1/woe-scorecard permanece para futuros cambios

---

## Reglas de oro para ramas

1. **Una rama por integrante/modelo**
   - No trabajar directamente en `main`
   - No crear sub-ramas de sub-ramas

2. **Nombres claros**
   ```
   P1/woe-scorecard          ✓ Bueno
   P1/feature                ✗ Muy genérico
   woe-scorecard             ✗ Falta [P#]
   P1-WOE-SCORECARD          ✗ Usar minúsculas
   ```

3. **Commits frecuentes**
   - Hacer commit cada cambio significativo
   - Máximo 1-2 notebooks modificados por commit

4. **Pull Requests antes de mergear**
   - NUNCA hacer push directo a `main`
   - Siempre abrir PR para revisión

5. **Actualizar antes de mergear**
   ```bash
   git fetch origin
   git rebase origin/main    # o merge si hay muchos commits
   ```

6. **Limpiar ramas locales**
   ```bash
   # Ver ramas locales
   git branch
   
   # Eliminar rama local después de mergear
   git branch -d P1/woe-scorecard
   
   # Eliminar rama remota
   git push origin -d P1/woe-scorecard
   ```

---

## Ejemplo completo: P1 implementa WOE

```bash
# 1. Clonar (primera vez)
git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
cd pi-bbva-mexico-ml

# 2. Crear rama
git checkout -b P1/woe-scorecard

# 3. Trabajar (multiple commits)
# - Editar notebook 01_P1_EDA_LR.ipynb
git add notebooks/P1_Logistic_Regression/01_P1_EDA_LR.ipynb
git commit -m "[P1] feat: análisis exploratorio del modelo LR"

# - Editar notebook 02_P1_WOE_Engineering.ipynb
git add notebooks/P1_Logistic_Regression/02_P1_WOE_Engineering.ipynb
git commit -m "[P1] feat: implementar WOE para todas las variables"

# 4. Actualizar con main (si hay cambios)
git fetch origin
git rebase origin/main

# 5. Push
git push -u origin P1/woe-scorecard

# 6. Crear PR en GitHub (interfaz web)
# - Base: main
# - Compare: P1/woe-scorecard
# - Descripción: "Implementar modelo de Regresión Logística con WOE y Scorecard"

# 7. P4 revisa y mergea en GitHub

# 8. Actualizar local (opcional)
git fetch origin
git checkout main
git pull origin main
```

---

## Conflictos en rebase

Si al hacer `git rebase origin/main` tienes conflictos:

```bash
# 1. Git te lo dirá
# On branch P1/woe-scorecard
# Unmerged paths:
#   both modified:   src/utils/evaluation.py

# 2. Abre el archivo y resuelve manualmente
# Las secciones conflictivas aparecen marcadas como:
# <<<<<<< HEAD
# tu cambio aquí
# =======
# cambio de main aquí
# >>>>>>> origin/main

# 3. Decide qué mantener, elimina los marcadores

# 4. Continúa el rebase
git add .
git rebase --continue

# 5. Si necesitas cancelar
git rebase --abort
```

---

## Merge vs Rebase

### Rebase (recomendado en este proyecto)
```bash
git rebase origin/main
```
- Pro: Historial linear y limpio
- Contra: Reescribe historia local (no usar si otros tienen tu rama)

### Merge (alternativa)
```bash
git merge origin/main
```
- Pro: Preserva toda la historia
- Contra: Commits de merge adicionales

Para este proyecto **usa rebase** porque cada rama es personal.

---

## Checklist antes de hacer Push

```
□ Código está funcionando (tests pasan)
□ Commits tienen buenos mensajes [P#] type: descripción
□ Rama está actualizada: git rebase origin/main
□ No hay conflictos sin resolver
□ Cambios están staged: git add .
□ Listo para push: git push origin Px/branch-name
```

---

## Soporte y preguntas

- **Problema con merge/rebase**: contactar a P4
- **Duda sobre rama**: ver este documento
- **Historial de cambios**: `git log --oneline`
