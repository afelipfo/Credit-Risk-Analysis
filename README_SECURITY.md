# Configuración de Seguridad - AWS Credentials

## ⚠️ IMPORTANTE: Configuración de Credenciales AWS

Este proyecto requiere credenciales de AWS para acceder a los datasets en S3. **NUNCA** incluyas credenciales reales en el código fuente.

### Configuración Local

1. **Copia el archivo de ejemplo:**
   ```bash
   cp .env.example .env
   ```

2. **Edita el archivo `.env` con tus credenciales reales:**
   ```
   AWS_ACCESS_KEY_ID=tu_access_key_real
   AWS_SECRET_ACCESS_KEY=tu_secret_key_real
   ```

3. **El archivo `.env` ya está incluido en `.gitignore`** para evitar que se suba accidentalmente a GitHub.

### Configuración en Producción

Para entornos de producción, configura las variables de entorno directamente:

```bash
export AWS_ACCESS_KEY_ID=tu_access_key_real
export AWS_SECRET_ACCESS_KEY=tu_secret_key_real
```

### Configuración con Docker

Si usas Docker, puedes pasar las variables de entorno:

```bash
docker run -e AWS_ACCESS_KEY_ID=tu_key -e AWS_SECRET_ACCESS_KEY=tu_secret tu_imagen
```

### Mejores Prácticas de Seguridad

1. **Usa AWS IAM Roles** cuando sea posible en lugar de Access Keys
2. **Rota las credenciales regularmente**
3. **Limita los permisos** a solo lo necesario (principio de menor privilegio)
4. **Nunca hardcodees credenciales** en el código
5. **Revisa regularmente los logs de acceso** de AWS

### Verificación

El código verificará automáticamente que las variables de entorno estén configuradas y mostrará un error claro si no están disponibles.
