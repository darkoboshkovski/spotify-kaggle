FROM node:14-slim as build-stage

WORKDIR /app
COPY package*.json /app/
RUN yarn install --production=true --frozen-lockfile --silent

COPY ./ /app/
RUN GENERATE_SOURCEMAP=false yarn build

FROM nginx:1.19
COPY --from=build-stage /app/build/ /usr/share/nginx/html
