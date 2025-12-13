FROM node:22-alpine AS builder
WORKDIR /app
ENV NPM_CONFIG_REGISTRY=https://registry.npmmirror.com/
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
