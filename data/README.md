# Datos

La fuente es la [API REData de Red Electrica](https://www.ree.es/en/datos/apidata), que expone widgets mediante peticiones GET y admite agregacion diaria. Este proyecto consulta `demanda/evolucion` para la demanda nacional diaria, de 2019 a 2025 por defecto.

- `data/raw/ree_demand_daily.json`: snapshot consolidado de las respuestas de API, con los valores sin normalizar.
- `data/raw/source_manifest.json`: fecha de descarga, URLs de cada peticion, recuentos y SHA-256.
- `data/processed/daily_demand.csv`: tabla diaria validada y enriquecida con calendario.
- `data/processed/model_features.csv`: tabla con lags y medias moviles para prediccion a un paso.

No se usan datos personales ni credenciales. El periodo puede configurarse con variables de entorno documentadas en `.env.example`.
