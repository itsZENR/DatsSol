#\!/bin/bash
set -e
until pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" 2>/dev/null; do
  echo "Waiting for postgres..."
  sleep 1
done
echo "Postgres is ready"
exec "$@"
