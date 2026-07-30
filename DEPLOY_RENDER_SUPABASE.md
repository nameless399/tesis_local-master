# Deploy Backend en Render + Supabase

## 1) Supabase
- Crea un proyecto en Supabase.
- Copia tu cadena de conexion Postgres (`Direct connection`).
- Asegurate de usar `sslmode=require`, por ejemplo:

```env
DATABASE_URL=postgresql://postgres:TU_PASSWORD@db.TU-PROYECTO.supabase.co:5432/postgres?sslmode=require
```

## 2) Render (Docker Web Service)
- En Render crea un servicio tipo `Web Service`.
- Elige `Docker` como runtime o importa el `render.yaml` del repo.
- Conecta este repositorio.
- Render usara el `Dockerfile` de la raiz.
- Si lo creas manualmente, no pongas `Build command` ni `Start command`.

## 3) Variables de entorno en Render
- `DATABASE_URL` = URL de Supabase con `sslmode=require`
- `MODELS_DIR` = `models_mix2` (o la carpeta real que uses)
- `POSE_WEIGHTS` = `yolo11s-pose.pt`
- `FUSION_W` = `0.50`
- `BOSS_CODE` = tu codigo para registro de jefe
- `CORS_ORIGINS` = URL de tu frontend (por ejemplo, dominio de Vercel)
- `CUDA_VISIBLE_DEVICES` = `-1`
- `TF_CPP_MIN_LOG_LEVEL` = `2`

## 4) Blueprint (opcional)
- Puedes usar `render.yaml` para crear el servicio con configuracion base.
- Los secretos (`DATABASE_URL`, `BOSS_CODE`) se cargan manualmente en Render.

## 5) Vercel
- En Vercel usa `Static Site` u `Other`, no `FastAPI`.
- Root directory: `.`.
- Build command: `npm run build`.
- Output directory: `static`.
- Variable de entorno: `API_BASE_URL` con la URL publica de Render, por ejemplo `https://tu-backend.onrender.com`.
- Si cambias `API_BASE_URL`, vuelve a desplegar Vercel para regenerar `config.js`.
