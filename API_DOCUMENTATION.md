# Documentación de la API de Predicción de Ventas

## Introducción

uvicorn main:app --host 0.0.0.0 --port 8000

Esta API proporciona funcionalidades para la predicción de ventas de equipos tecnológicos, el análisis de los productos más vendidos y la visualización de pronósticos. Está diseñada para ser consumida por aplicaciones frontend, permitiendo integrar capacidades de análisis predictivo y visualización de datos.

**URL Base:** `http://0.0.0.0:8000` (Esta URL puede variar dependiendo de dónde se despliegue la API).

---

## Endpoints

### 1. Predicción de Ventas Futuras

*   **Endpoint:** `/predict/sales/{days}`
*   **Descripción:** Predice las ventas totales para un número específico de días futuros utilizando un modelo Prophet entrenado.
*   **Método HTTP:** `GET`
*   **Ruta:** `/predict/sales/{days}`

#### Parámetros de Ruta

*   **`days`**
    *   **Tipo:** `integer`
    *   **Requerido:** Sí
    *   **Descripción:** El número de días futuros para los que se desea generar la predicción.
    *   **Validación:** Debe ser un entero positivo (`> 0`).

#### Ejemplo de Solicitud

```bash
curl http://0.0.0.0:8000/predict/sales/7
```

#### Ejemplo de Respuesta (JSON)

```json
{
  "forecast": [
    {
      "ds": "2019-10-02",
      "yhat": 98476.42973850515,
      "yhat_lower": 86568.617277573,
      "yhat_upper": 108789.3893281644
    },
    {
      "ds": "2019-10-03",
      "yhat": 98882.20674868822,
      "yhat_lower": 87953.17519100153,
      "yhat_upper": 108810.43340216517
    },
    // ... más días ...
    {
      "ds": "2019-10-08",
      "yhat": 124956.88252899682,
      "yhat_lower": 113438.81776629829,
      "yhat_upper": 135572.34380933503
    }
  ],
  "metrics": null
}
```

#### Campos de la Respuesta

*   **`forecast`** (array de objetos): Una lista de objetos de predicción, uno por cada día futuro.
    *   **`ds`** (string): La fecha de la predicción en formato `YYYY-MM-DD`.
    *   **`yhat`** (float): El valor central de la predicción de ventas para esa fecha.
    *   **`yhat_lower`** (float): El límite inferior del intervalo de confianza del 80% para la predicción de ventas.
    *   **`yhat_upper`** (float): El límite superior del intervalo de confianza del 80% para la predicción de ventas.
*   **`metrics`** (null): Actualmente no se devuelven métricas de evaluación en este endpoint.

#### Uso para Frontend

Un frontend puede consumir este endpoint para mostrar una tabla o un gráfico de las ventas futuras esperadas. Los valores `yhat_lower` y `yhat_upper` son útiles para visualizar la incertidumbre del pronóstico, por ejemplo, como una banda sombreada en un gráfico de líneas.

---

### 2. Análisis de Productos Más Vendidos

*   **Endpoint:** `/analysis/bestsellers/{top_n}`
*   **Descripción:** Identifica y lista los `top_n` productos que han generado la mayor cantidad de ingresos por ventas en el período histórico analizado.
*   **Método HTTP:** `GET`
*   **Ruta:** `/analysis/bestsellers/{top_n}`

#### Parámetros de Ruta

*   **`top_n`**
    *   **Tipo:** `integer`
    *   **Requerido:** Sí
    *   **Descripción:** El número de productos top que se desean recuperar.
    *   **Validación:** Debe ser un entero positivo (`> 0`).

#### Ejemplo de Solicitud

```bash
curl http://0.0.0.0:8000/analysis/bestsellers/3
```

#### Ejemplo de Respuesta (JSON)

```json
{
  "bestsellers": [
    {
      "product": "Macbook Pro Laptop",
      "total_sales_revenue": 8037600.0
    },
    {
      "product": "iPhone",
      "total_sales_revenue": 4794300.0
    },
    {
      "product": "ThinkPad Laptop",
      "total_sales_revenue": 4129958.7
    }
  ]
}
```

#### Campos de la Respuesta

*   **`bestsellers`** (array de objetos): Una lista de objetos, cada uno representando un producto top.
    *   **`product`** (string): El nombre del producto.
    *   **`total_sales_revenue`** (float): El ingreso total generado por este producto.

#### Uso para Frontend

Este endpoint es ideal para mostrar un ranking de productos en un dashboard, una sección de "Productos Populares" o para informar decisiones de inventario y campañas de marketing.

---

### 3. Visualización del Pronóstico (PNG)

*   **Endpoint:** `/chart/forecast/{days}`
*   **Descripción:** Genera un gráfico de la previsión de ventas y lo devuelve directamente como una imagen PNG.
*   **Método HTTP:** `GET`
*   **Ruta:** `/chart/forecast/{days}`

#### Parámetros de Ruta

*   **`days`**
    *   **Tipo:** `integer`
    *   **Requerido:** Sí
    *   **Descripción:** El número de días futuros para los que se desea generar el gráfico de previsión.
    *   **Validación:** Debe ser un entero positivo (`> 0`).

#### Ejemplo de Solicitud

```bash
curl -o forecast_chart.png http://0.0.0.0:8000/chart/forecast/30
```

#### Ejemplo de Respuesta

La respuesta es una imagen PNG binaria. El frontend puede mostrar esta imagen directamente en una etiqueta `<img>`.

#### Uso para Frontend

Este endpoint es útil cuando el frontend necesita mostrar una imagen de gráfico directamente, por ejemplo, en un informe o un dashboard donde la imagen se carga como un recurso independiente.

---

### 4. Visualización del Pronóstico (Base64)

*   **Endpoint:** `/chart/forecast_base64`
*   **Descripción:** Genera un gráfico de la previsión de ventas y lo devuelve como una cadena codificada en Base64 dentro de un objeto JSON.
*   **Método HTTP:** `GET`
*   **Ruta:** `/chart/forecast_base64`

#### Parámetros de Consulta (Query Parameters)

*   **`days`**
    *   **Tipo:** `integer`
    *   **Requerido:** No (tiene un valor por defecto de 90 días si no se especifica).
    *   **Descripción:** El número de días futuros para los que se desea generar el gráfico de previsión.
    *   **Validación:** Debe ser un entero positivo (`> 0`).

#### Ejemplo de Solicitud

```bash
curl "http://0.0.0.0:8000/chart/forecast_base64?days=30"
```
O sin especificar `days` para usar el valor por defecto:
```bash
curl "http://0.0.0.0:8000/chart/forecast_base64"
```

#### Ejemplo de Respuesta (JSON)

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAB4AAAAQ4CAYAAADo0W+LAAAgAElEQVR4nOzdd3xU1f/A8e+999733nvv...",
  "media_type": "image/png",
  "metrics": null
}
```
(La cadena `image_base64` será mucho más larga en una respuesta real).

#### Campos de la Respuesta

*   **`image_base64`** (string): La imagen del gráfico codificada en formato Base64.
*   **`media_type`** (string): El tipo MIME de la imagen, siempre "image/png".
*   **`metrics`** (null): Actualmente no se devuelven métricas de evaluación en este endpoint.

#### Uso para Frontend

Este endpoint es ideal para aplicaciones frontend (especialmente Single Page Applications o SPAs) que necesitan incrustar la imagen del gráfico directamente en el HTML/CSS/JavaScript sin hacer una solicitud HTTP separada. La cadena Base64 puede asignarse directamente al atributo `src` de una etiqueta `<img>` (precedida por `data:image/png;base64,`).

---

## Consideraciones Adicionales para el Frontend

*   **Manejo de Errores:** La API devolverá códigos de estado HTTP estándar en caso de errores:
    *   `400 Bad Request`: Si los parámetros de entrada no cumplen con las validaciones (ej. `days` no es un entero positivo).
    *   `404 Not Found`: Si la ruta solicitada no existe.
    *   `500 Internal Server Error`: Si ocurre un error inesperado en el servidor (ej. el modelo no pudo cargarse al inicio).
    *   `503 Service Unavailable`: Si el modelo no está cargado o los datos no han sido procesados (esto ocurre si la API no se inicializó correctamente).
*   **CORS:** Asegúrate de que la API esté configurada para permitir solicitudes desde el dominio de tu frontend si se ejecutan en dominios diferentes. FastAPI soporta CORS a través de `CORSMiddleware`.
*   **Rendimiento:** La generación de gráficos puede ser intensiva en recursos. Considera implementar caché en el frontend o en un proxy si la misma imagen se solicita con frecuencia.
*   **Autenticación/Autorización:** Actualmente, la API no implementa autenticación. Para entornos de producción, se recomienda añadir mecanismos de seguridad adecuados.
*   **Reinicio del Servidor:** Cualquier cambio en el código Python de la API requiere un reinicio del servidor FastAPI para que los cambios surtan efecto.

---
