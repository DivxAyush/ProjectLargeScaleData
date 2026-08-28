# Integration Tests

This directory contains tests that require external infrastructure (e.g., a real MongoDB instance).

These tests are excluded from the normal test suite run unless the required environment variables are present.

## Running MongoDB Integration Tests

1. Start a local MongoDB instance (e.g., via Docker):
   ```bash
   docker run -d -p 27017:27017 --name mongo-test mongo:6
   ```

2. Run the integration tests:
   ```bash
   MONGODB_URI="mongodb://localhost:27017" MONGODB_DB_NAME="mili_test" pytest tests/integration/ -v
   ```
