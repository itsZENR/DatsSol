FROM golang:1.25-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /collector ./cmd/collector/
RUN CGO_ENABLED=0 go build -o /player ./cmd/player/
RUN CGO_ENABLED=0 go build -o /dashboard ./cmd/dashboard/

FROM alpine:3.21
RUN apk add --no-cache bash postgresql-client
COPY --from=builder /collector /usr/local/bin/collector
COPY --from=builder /player /usr/local/bin/player
COPY --from=builder /dashboard /usr/local/bin/dashboard
COPY migrations/ /app/migrations/
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh
