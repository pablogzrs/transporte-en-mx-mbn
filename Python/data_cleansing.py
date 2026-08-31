"""
--------------------------------------------------------------------------------
DECISIONES DE DISEÑO Y SU JUSTIFICACION
--------------------------------------------------------------------------------

(!) Si bien es cierto toda esta sección de decisión de diseño y sus justificaciones son de nuestra autoría,
gran parte del código que logra plasmar ello en nuestra limpieza lo generó la IA Claude en base a los siguientes criterios que establecimos al analizar la base de datos
y las preguntas que necesitamos responder. 

I. A considerarse 

(1) ENCODING
    El archivo no es UTF-8. Se usa cp1252 y no latin-1.

(2) dtype=str
    TODOS los codigos de esta encuesta son categoricos aunque parezcan numeros.

(3) CODIGOS DE NO-RESPUESTA: NO SON UNIFORMES
    df.isna().sum() devuelve cero en el archivo crudo porque los faltantes estan
    codificados como numeros validos. Y el codigo cambia segun la variable.

    Ademas, el -1 aparece en varias columnas como "no se le pregunto" (salto de
    cuestionario). Se trata como faltante en todas.
    
II. Criterios de selección, eliminación y agrupación

(4) VARIABLES ELIMINADAS 

No se alinean con las preguntas objetivo: Motivo de uso, variables que involucren caminar, preguntas generales sobre automóvil SALVO si la persona cuenta con uno (p6),
preguntas sobre conduccion, medios para trasladarse a otros estados/ países, dimensiones de vivienda, cuantas personas hay en una vivienda e información derivada de ellas 
por el informante (entrevistado), alfabetismo, estad civil, cómo hace para sostenerse, ocupación (específica, no empleado/ desempleado), cómo recibe remuneración económicoa y 
generalidades de contrato, variables específicas para discapacitados, accidentes de tránsito, estrato, municipio, localidad. Estado se usa para la creación de una nueva 
variable, para luego ser descartada. 


(5) Corrección de escala en variable de interés ing_fam:
    * EL 0 ES NO-RESPUESTA, NO INGRESO CERO
    El diccionario de ing_fam etiqueta el codigo 1 como "NS/NC" y del 2 al 8
    como rangos de salarios minimos. En los datos hay 0 y del 2 al 8, sin
    ningun 1.


(6) AGRUPACION DE MODOS DE TRANSPORTE
    Los 22 modos se colapsan a 4 familias. Criterios:
      - TRANSPORTE MASIVO: tren urbano, BRT, eléctrico. Tipo de tramsporte que sólo se encuentra en las grandes metrópolis de la república (es decir, ZMVM, ZMM, ZMG)
      - CONSECIONADO: camión, microbus, colectivo (combi, camioneta, minivan). Tipo de transporte citadino mas no masivo. 
      - PARTICULAR: auto, motocicleta. Transporte propio y no público
      - OTRO: tren foráneo, autobús escolar/ foráneo, avión, tractor, tráiler, tracción animal, animal, helicóptero, embarcaciones, taxi, bicitaxi, bicicleta, patines, y 
      NINGUNO (muy importante la inclusión de este ultimo). Es decir, métodos de transporte esporádicos.

(7) CRITERIO DE INFRAESTRUCTURA FERROVIARIA URBANA (tiene_sistema_tren)
    Los 63 usuarios cotidianos de tren urbano estan en Edomex (25), CDMX (20),
    Nuevo Leon (17) y Jalisco (1). Fuera de esas entidades este tipo de transporte no existe,
    asi que comparar percepcion de usuarios de tren contra usuarios de cualquier otra familia antes definida
    sin condicionar mezclaria "usar metro" con "vivir donde hay metro".
    Se deriva de estado (misma que después es eliminada) ya que son solo 4 estados los que cuentan con esta infraestructura y considerar todos los estados
    haría más grande y por lo tanto más computacionalmente costoso el modelo de manera innecesaria

(8) MODO PRINCIPAL Y DESEMPATE
    p1a es frecuencia en 3 niveles, asi que hay empates masivos:
    216 personas usan 2 familias cotidianamente, 48 usan 3, 9 usan 4, y 155 usan camion y
    colectivo a la vez. 242 no reportan ningun modo cotidiano.

    Regla de desempate, en orden:
      1. Entre las familias usadas cotidianamente, gana aquella cuyo motivo de
         viaje (p1b) sea obligado: trabajo (1), clases (3) o relacionado con el
         trabajo (9). 
      2. Si persiste el empate, jerarquia por compromiso estructural:
         particular > masivo > concesionado > otros.
      3. Sin ningun modo cotidiano -> "otro" (242 casos, identificables con
         n_modos_cotidianos == 0). 
      
  
--------------------------------------------------------------------------------
"""

# Rutas relativas: el script asume ejecutarse desde la carpeta Python/ del
# proyecto (por eso ../Data). Entrada: enmt_unam.csv (1191 x 698).
# Salida: modelo_a.csv (1117 x 17), entra directo a hc() de bnlearn.


import os

import pandas as pd
import numpy as np

RUTA_ENTRADA = "../Data/enmt_unam.csv"
DIR_SALIDA   = "../Data"


# =============================================================================
# 1. CATALOGO DE MODOS Y FAMILIAS
# =============================================================================

# Familias por regimen operativo. Ver justificacion (6).
FAMILIAS = {
    "masivo":       [2, 7, 3],    # tren urbano, BRT, electrico
    "concesionado": [4, 5],       # camion/microbus, colectivo (combi, minivan)
    "particular":   [12, 15],     # automovil, motocicleta
    "otro":         [1, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22],
                    # tren foraneo, bus foraneo, taxi, bicitaxi, escolar, avion,
                    # tractor, trailer, bici, patines, traccion animal, animal,
                    # helicoptero, embarcaciones
}

# Nodos del DAG. Fuera quedan: con1 (identificador; va como indice del CSV),
# n_modos_cotidianos (diagnostico, funcion determinista de las uso_*) y las
# uso_* (redundancia determinista con modo_usado: conocerlas fija el modo).
VARS_MODELO = [
    "region", "tam_loc", "tiene_sistema_tren",
    "sexo", "edad", "escolaridad", "ingreso_familiar", "condicion_actividad",
    "tiene_auto", "modo_usado",
    "tp_eficiente", "tp_rapido", "tp_barato", "tp_seguro", "tp_comodo",
    "contamina_su_modo", "contamina_aire",
]
VARS_CALIF = ["calif_comodidad", "calif_seguridad"]

# Desempate paso 2. Ver justificacion (8).
JERARQUIA = ["particular", "masivo", "concesionado", "otro"]

# Motivos de viaje obligados en p1b. Desempate paso 1.
MOTIVOS_OBLIGADOS = {"1", "3", "9"}   # trabajo, clases, relacionado con trabajo

# Entidades con sistema ferroviario urbano. Ver justificacion (7).
EDOS_CON_TREN = {"9", "15", "19", "14"}   # CDMX, Edomex, Nuevo Leon, Jalisco


# =============================================================================
# 2. COLUMNAS A LEER
# =============================================================================

P1A = [f"p1a_{i}" for i in range(1, 23)]                        # frecuencia de uso
P1B = [f"p1b_{i}" for i in range(1, 23)]                        # motivo del viaje
P1C = [f"p1c_{m}_{a}" for m in range(1, 23) for a in (2, 3)]   # seguridad, comodidad
P17 = [f"p17_{i}" for i in range(1, 6)]

COLUMNAS = (
    ["con1", "edo", "Region", "Tam_loc"]
    + ["sexo", "edad_1", "escol", "ing_fam", "cond_act"]
    + P1A + P1B + ["p6"] + P1C
    + P17 + ["p19", "p20"]
)


# =============================================================================
# 3. CODIGOS DE NO-RESPUESTA POR COLUMNA. Ver justificacion (3).
# =============================================================================

NA_MAP = {}
for c in P1A:                            NA_MAP[c] = ["8", "9", "-1"]
for c in P1B:                            NA_MAP[c] = ["97", "98", "99", "-1"]
for c in P1C:                            NA_MAP[c] = ["97", "98", "99", "-1"]
for c in ["p6", "p19", "p20"]:    NA_MAP[c] = ["98", "99", "-1"]
NA_MAP["escol"]    = ["8", "9"]
NA_MAP["cond_act"] = ["-1"]
NA_MAP["ing_fam"]  = ["-1", "988888", "999999"]   # el 0 NO va aqui: ver (5)


# =============================================================================
# 4. ETIQUETAS
# =============================================================================

ETIQUETAS = {
    "sexo":     {"1": "hombre", "2": "mujer"},
    "edad_1":   {"2": "18-24", "3": "25-34", "4": "35-44",
                 "5": "45-54", "6": "55-64", "7": "65+"},
    "escol":    {"1": "ninguna", "2": "primaria", "3": "secundaria",
                 "4": "preparatoria", "5": "universidad"},
    "ing_fam":  {"0": "no_declaro",
                 "2": "menos_1sm", "3": "1-2sm", "4": "2-3sm", "5": "3-4sm",
                 "6": "4-5sm", "7": "5-6sm", "8": "mas_6sm"},
    "cond_act": {"1": "ocupado", "2": "no_ocupado"},
    "Region":   {"1": "region_1", "2": "region_2",
                 "3": "region_3", "4": "region_4"},
    "Tam_loc":  {"1": "100mil_mas", "2": "15mil_99999",
                 "3": "2500_14999", "4": "menos_2500"},
    "p6":       {"1": "si", "2": "no"},
    "p19":      {"1": "no_contamina", "2": "poco", "3": "algo", "4": "mucho"},
    "p20":      {"1": "nada", "2": "poco", "3": "algo", "4": "mucho"},
    "p17_1":    {"1": "eficiente", "2": "ineficiente"},
    "p17_2":    {"1": "rapido",    "2": "lento"},
    "p17_3":    {"1": "barato",    "2": "caro"},
    "p17_4":    {"1": "seguro",    "2": "inseguro"},
    "p17_5":    {"1": "comodo",    "2": "incomodo"},
}

RENOMBRES = {
    "Region": "region", "Tam_loc": "tam_loc",
    "p6": "tiene_auto", "p19": "contamina_su_modo",
    "p20": "contamina_aire", "p17_1": "tp_eficiente", "p17_2": "tp_rapido",
    "p17_3": "tp_barato", "p17_4": "tp_seguro", "p17_5": "tp_comodo",
    "edad_1": "edad", "escol": "escolaridad", "ing_fam": "ingreso_familiar",
    "cond_act": "condicion_actividad",
}


# =============================================================================
# 5. CARGA
# =============================================================================

def cargar(ruta=RUTA_ENTRADA):
    """Lee el CSV crudo con el encoding correcto y los faltantes por columna."""
    df = pd.read_csv(
        ruta,
        encoding="cp1252",
        dtype=str,
        low_memory=False,
        usecols=lambda c: c in COLUMNAS,
        na_values=NA_MAP,
    )
    return df[[c for c in COLUMNAS if c in df.columns]]


# =============================================================================
# 6. DERIVACION DE VARIABLES
# =============================================================================

def agregar_uso_por_familia(df):
    """
    Colapsa los 22 modos a las 4 familias (justificacion 6). El uso de una
    familia es el MINIMO de sus columnas p1a, porque 1 cotidiano < 2 ocasional
    < 3 nunca. min ignora NaN; todo-NaN da NaN.
    """
    for fam, modos in FAMILIAS.items():
        cols = [f"p1a_{m}" for m in modos]
        df[f"uso_{fam}"] = (df[cols].astype(float).min(axis=1)
                            .map({1.0: "cotidiano", 2.0: "ocasional", 3.0: "nunca"}))
    return df


def _motivo_obligado_por_familia(df, fam):
    """True si algun modo COTIDIANO de la familia tuvo motivo obligado
    (p1b in MOTIVOS_OBLIGADOS). Solo modos cotidianos: justificacion (8)."""
    flags = [
        (df[f"p1a_{m}"] == "1") & df[f"p1b_{m}"].isin(MOTIVOS_OBLIGADOS)
        for m in FAMILIAS[fam] if f"p1b_{m}" in df.columns
    ]
    if not flags:
        return pd.Series(False, index=df.index)
    return pd.concat(flags, axis=1).any(axis=1)


def agregar_modo_usado(df):
    """
    Familia principal por persona, con la regla de la justificacion (8):
    cotidianas -> filtro de motivo obligado -> JERARQUIA -> sin nada = "otro".
    Crea n_modos_cotidianos para reportar cuantos casos requirieron desempate.
    """
    cotidiano = pd.DataFrame(
        {fam: df[f"uso_{fam}"].eq("cotidiano") for fam in FAMILIAS},
        index=df.index,
    )
    obligado = pd.DataFrame(
        {fam: _motivo_obligado_por_familia(df, fam) for fam in FAMILIAS},
        index=df.index,
    )

    df["n_modos_cotidianos"] = cotidiano.sum(axis=1)

    def elegir(i):
        candidatas = [f for f in JERARQUIA if cotidiano.at[i, f]]
        if not candidatas:
            return "otro"
        if len(candidatas) > 1:
            con_motivo = [f for f in candidatas if obligado.at[i, f]]
            if con_motivo:
                candidatas = con_motivo
        return candidatas[0]        # JERARQUIA ya define el orden

    df["modo_usado"] = [elegir(i) for i in df.index]
    return df


def agregar_bandera_tren(df):
    """Entidad con sistema ferroviario urbano (justificacion 7). Se deriva
    aqui porque edo no sobrevive a la seleccion final."""
    df["tiene_sistema_tren"] = np.where(
        df["edo"].isin(EDOS_CON_TREN), "si", "no"
    )
    return df


def agregar_calificacion_modo_usado(df):
    """
    Calificacion p1c (seguridad, comodidad) del modo cotidiano de la familia
    principal; promedio si usa varios de la familia. Discretizada en bajo 0-5 /
    medio 6-7 / alto 8-10 (la distribucion carga a la derecha; terciles no son
    interpretables). NaN para quien no usa nada cotidianamente.
    OJO: cada quien califica SU modo -- no es escala comun entre personas; para
    eso esta tp_comodo.
    """
    for asp, nombre in [(3, "comodidad"), (2, "seguridad")]:
        vals = []
        for i in df.index:
            fam = df.at[i, "modo_usado"]
            cols = [f"p1c_{m}_{asp}" for m in FAMILIAS[fam]
                    if df.at[i, f"p1a_{m}"] == "1"]
            v = [float(df.at[i, c]) for c in cols
                 if c in df.columns and pd.notna(df.at[i, c])]
            vals.append(np.mean(v) if v else np.nan)
        df[f"calif_{nombre}"] = pd.cut(
            pd.Series(vals, index=df.index),
            bins=[-0.1, 5, 7, 10], labels=["bajo", "medio", "alto"]
        ).astype(object)
    return df


# =============================================================================
# 7. PIPELINE
# =============================================================================

def limpiar(ruta=RUTA_ENTRADA):
    df = cargar(ruta)
    df = agregar_uso_por_familia(df)
    df = agregar_modo_usado(df)
    df = agregar_bandera_tren(df)
    df = agregar_calificacion_modo_usado(df)

    # Etiquetado de las variables que se quedan como nodos.
    for col, mapa in ETIQUETAS.items():
        if col in df.columns:
            df[col] = df[col].map(mapa)

    df = df.rename(columns=RENOMBRES)

    # Columnas finales: se tiran edo (ya derivada), p1a_*, p1b_* y p1c_* crudas.
    finales = (
        ["con1", "region", "tam_loc", "tiene_sistema_tren"]
        + ["sexo", "edad", "escolaridad", "ingreso_familiar", "condicion_actividad"]
        + ["tiene_auto", "modo_usado", "n_modos_cotidianos"]
        + [f"uso_{f}" for f in FAMILIAS]
        + ["tp_eficiente", "tp_rapido", "tp_barato", "tp_seguro", "tp_comodo"]
        + ["contamina_su_modo", "contamina_aire"]
        + [c for c in df.columns if c.startswith("calif_")]
    )
    return df[[c for c in finales if c in df.columns]]


def exportar_modelos(df, carpeta=DIR_SALIDA):
    """
    Escribe modelo_a.csv listo para hc() en R. con1 va como indice del CSV
    (row.names=1 en R): trazable pero invisible para bnlearn. complete.cases
    se aplica aqui, no en R, para que el n quede documentado en el pipeline.

    Cortes de robustez, si se necesitan, se agregan al diccionario salidas:
      "modelo_b.csv":         df[["con1"] + VARS_MODELO + VARS_CALIF].dropna()
          n~880; segundo instrumento de las preguntas 1 y 2; excluye por
          construccion a quien no usa ningun modo cotidiano.
      "modelo_a_robusto.csv": df.loc[df["n_modos_cotidianos"] == 1,
                                     ["con1"] + VARS_MODELO].dropna()
          solo asignaciones sin desempate (la regla aplico en el 43%); sirve
          para preguntas 2-4, no para la 1 (deja ~14 casos de masivo).
    """
    salidas = {
        "modelo_a.csv": df[["con1"] + VARS_MODELO].dropna(),
    }
    for nombre, sub in salidas.items():
        ruta = os.path.join(carpeta, nombre)
        sub.set_index("con1").to_csv(ruta, encoding="utf-8")
        masivo = (sub["modo_usado"] == "masivo").sum()
        print(f"  {nombre:16s} n={len(sub):5d}  vars={sub.shape[1]-1:3d}  "
              f"masivo={masivo:3d}")
    return salidas


def reporte(df):
    """Diagnostico para pegar en la seccion de datos del articulo."""
    print(f"Filas: {len(df)}   Columnas: {df.shape[1]}\n")
    print("--- no-respuesta por columna ---")
    na = df.isna().sum()
    print(na[na > 0].sort_values(ascending=False).to_string() or "(ninguna)")
    print(f"\nColumnas sin ningun faltante: {(na == 0).sum()} de {df.shape[1]}")
    print("\n--- modo usado ---")
    print(df["modo_usado"].value_counts().to_string())
    print("\n--- casos que requirieron desempate ---")
    print(f"con 2+ modos cotidianos: {(df['n_modos_cotidianos'] > 1).sum()}")
    print(f"sin ningun modo cotidiano: {(df['n_modos_cotidianos'] == 0).sum()}")
    print("\n--- variables clave de cada pregunta ---")
    for c in ["tp_comodo", "tp_seguro", "tp_eficiente",
              "contamina_su_modo", "tiene_auto", "tam_loc"]:
        print(f"{c:20s} {df[c].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    df = limpiar()
    reporte(df)
    print("\n--- dataset de modelado ---")
    exportar_modelos(df)