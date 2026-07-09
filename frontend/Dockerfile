# --- estágio 1: build do Angular ----------------------------------------
FROM node:24-slim AS builder
WORKDIR /app

# manifests primeiro: camada de dependências só rebuilda se elas mudarem
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npx ng build --configuration production

# --- estágio 2: nginx servindo o estático --------------------------------
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist/credit-risk-front/browser /usr/share/nginx/html
EXPOSE 80
