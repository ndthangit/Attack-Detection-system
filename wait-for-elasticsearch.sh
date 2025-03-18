#!/bin/sh
echo "Waiting for Elasticsearch to be ready..."
until curl -s http://elasticsearch:9200/_cluster/health | grep -q '"status":"green"\|"status":"yellow"'; do
  sleep 60
  echo "Elasticsearch is not ready yet..."
done
echo "Elasticsearch is ready!"
exec "$@"
