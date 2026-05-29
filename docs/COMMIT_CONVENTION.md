# Convención de Commits - PI BBVA México

## Formato de commit

```
[P#] type: descripción corta

Descripción detallada (opcional)
- Punto 1
- Punto 2
```

### Componentes

1. **[P#]**: Identificador del integrante (P1, P2, P3, P4)
2. **type**: Tipo de cambio (ver sección siguiente)
3. **descripción corta**: Una línea, máximo 50 caracteres, imperativo

## Tipos de cambios permitidos

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `feat` | Nueva funcionalidad | `[P1] feat: implementar WOE para variables continuas` |
| `fix` | Corrección de bugs | `[P2] fix: corregir cálculo de AUROC en evaluación` |
| `refactor` | Mejora de código sin cambiar funcionalidad | `[P3] refactor: modularizar función de splits` |
| `docs` | Documentación | `[P4] docs: agregar instrucciones de instalación` |
| `test` | Tests o validación | `[P1] test: validar WOE calculation` |
| `chore` | Tareas de configuración (requirements, .gitignore) | `[P4] chore: agregar librerías a requirements.txt` |
| `style` | Cambios de formato, espacios, etc. | `[P2] style: formatear código con black` |

## Ejemplos válidos

### Buen ejemplo - Feature (M1 - P1)
```
[P1] feat: implementar WOE y scorecard para regresión logística

- Crear función para binning automático
- Calcular Weight of Evidence por variable
- Generar scorecard con puntos
- Validar en train, test y OOT
```

### Buen ejemplo - Fix (M2 - P2)
```
[P2] fix: corregir cálculo del KS statistic en evaluación

Problema: tpr - fpr generaba valores negativos
Solución: usar abs() y calcular correctamente el máximo
```

### Buen ejemplo - Refactor (M3 - P3)
```
[P3] refactor: extraer lógica de splits a módulo separado

Movido: scripts/splits.py → src/utils/splits.py
Beneficio: reutilizable por todos los integrantes
```

### Buen ejemplo - Docs (M4 - P4)
```
[P4] docs: agregar plantillas de notebooks

- Plantilla para P1: 01_P1_EDA_LR.ipynb
- Plantilla para P2: 01_P2_EDA_XGB.ipynb
- Plantilla para P3: 01_P3_EDA_RF.ipynb
- Plantilla para P4: 01_P4_EDA_NN.ipynb
```

## Ejemplos INCORRECTOS ❌

```
Updated model              # Sin [P#] ni type
[P1] Fixed stuff           # Sin descripción clara
[P2] feat WOE implementation # Punto y dos puntos faltantes
P1 feat: hacer el modelo   # Falta corchete en [P1]
[P1] FEAT: todo            # type debe estar en minúscula
```

## Reglas de oro

1. **Siempre incluir [P#]** al inicio del mensaje
2. **Usar imperativo**: "implementar", "agregar", "corregir" (no "implementado", "agregó", "corrigió")
3. **Descripción corta clara**: máximo 50 caracteres
4. **Commits atómicos**: un cambio principal por commit
5. **Descripción detallada (opcional)**: cuando el cambio es complejo, agregar puntos clave

## Malas prácticas a evitar

```
# ❌ EVITAR
[P1] Updated files
[P2] WIP: working on tuning
[P3] fixed typo in variable name (esto es muy menor)
[P4] [wip] neural network
[P1] feat implemented        # Faltan dos puntos
```

## Buen flujo de commits

```bash
# Trabajar en rama personal
git checkout -b P1/woe-scorecard

# Hacer cambios
# ...

# Commit 1
git commit -m "[P1] feat: crear función de binning WOE"

# Cambios adicionales
# ...

# Commit 2
git commit -m "[P1] feat: generar scorecard con puntos"

# Cambios finales
# ...

# Commit 3
git commit -m "[P1] test: validar WOE en train/test/OOT"

# Push
git push origin P1/woe-scorecard
```

## Convención de rama

Las ramas deben seguir el patrón:
```
[P#]/feature-description

Ejemplos:
- P1/woe-scorecard
- P2/xgboost-tuning
- P3/random-forest-features
- P4/neural-network
- P1/fix-scorecard-calculation
```

## Verificación antes de hacer commit

```bash
# Ver qué vas a commitear
git status
git diff

# Asegurar que el código está limpio
python -m black src/
python -m flake8 src/

# Hacer commit con buen mensaje
git commit -m "[P#] type: descripción"
```

## Referencias

Para más detalles, consultar:
- Conventional Commits: https://www.conventionalcommits.org/
- Este proyecto: README.md § Convención de commits
