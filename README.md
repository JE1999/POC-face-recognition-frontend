# POC de reconocimiento facial, anti-spoof y control de entorno

Aplicación experimental en Next.js que compara una fotografía con la cámara, clasifica la presentación facial, busca teléfonos en el encuadre y ejecuta desafíos de movimiento en el navegador. No incluye backend, no registra asistencia y no persiste datos biométricos.

## Ejecutar

Requisitos: Node.js 20 o superior, pnpm 10 y un navegador moderno con WebGL.

```bash
pnpm install
pnpm dev
```

Abre [http://localhost:3000](http://localhost:3000). El acceso a cámara requiere `localhost` o HTTPS.

Validaciones disponibles:

```bash
pnpm typecheck
pnpm lint
pnpm build
```

## Flujo de prueba

1. Carga un JPG, PNG o WebP frontal con una sola persona.
2. Espera el estado **Referencia válida**.
3. Inicia la cámara y permite el acceso solicitado por el navegador.
4. Espera tres lecturas concordantes de identidad.
5. Espera tres lecturas consecutivas de MiniFASNetV2 clasificadas como rostro real.
6. Confirma que MediaPipe no detecta un teléfono en el encuadre.
7. Inicia la verificación activa y sigue la secuencia aleatoria.
8. Consulta por separado identidad, anti-spoof, entorno, verificación activa y resultado general.

Para probar errores, usa una imagen sin rostro, una foto grupal, bloquea el permiso de cámara, presenta otra persona o sal del encuadre durante un desafío. En móvil conviene cerrar otras aplicaciones que usen la cámara.

## Modelos y arquitectura

- `@vladmandic/face-api`: `TinyFaceDetector` a 320 px, landmarks de 68 puntos y descriptor de reconocimiento. La cámara se analiza de forma espaciada; no se procesa cada frame.
- MiniFASNetV2: clasificador ONNX de tres clases sobre un recorte BGR de 80×80 ampliado 2.7× alrededor de la caja facial. Se ejecuta con `onnxruntime-web`; la clase 1 es el rostro real y la decisión exige tres cuadros consecutivos concordantes.
- `@mediapipe/tasks-vision`: Face Landmarker para landmarks, blendshapes y movimientos; Object Detector con EfficientDet-Lite0 y filtro `cell phone` para revisar el entorno.
- La detección de teléfonos analiza un cuadro cada 300 ms. Bloquea inmediatamente desde `0.45` de confianza o al superar el umbral sensible inicial `0.22` en 2 de las últimas 4 lecturas; se libera después de 4 lecturas negativas.
- Los modelos y binarios WASM están en `public/models` y `public/wasm`; no hay CDN en tiempo de ejecución.
- La decisión facial usa 3 votos coincidentes en una ventana de 4 lecturas. El umbral inicial `0.50` es editable y debe calibrarse para cada entorno.
- El parpadeo exige la transición abierto → cerrado → abierto. Los giros se comparan contra la orientación capturada al iniciar cada instrucción.
- Los módulos de visión, cámara, modelos y máquina de estados están separados en `src/lib` y `src/hooks`.

## Privacidad, límites y camino a producción

Fotografías, video y descriptores permanecen en memoria y se eliminan al reemplazar la referencia, detener la cámara o cerrar la página. No se usa `localStorage`, cookies, base de datos ni telemetría biométrica.

Esta POC no es una prueba de vida certificada. MiniFASNetV2 reduce algunos ataques de presentación, pero su puntaje no está calibrado para este entorno y puede degradarse con iluminación, desenfoque o ataques no representados en su entrenamiento. EfficientDet-Lite0 puede omitir teléfonos pequeños, tapados o vistos desde ángulos difíciles, y detectar uno no demuestra que esté grabando. Antes de producción se necesita calibración con datos representativos, evaluación de sesgo, consentimiento, revisión legal y pruebas en los dispositivos objetivo.

`TinyFaceDetector` favorece velocidad y tamaño sobre precisión. Para escenarios difíciles se debe evaluar un detector más preciso, calidad de captura, umbrales por dispositivo y ejecución en worker para evitar bloquear la interfaz.
