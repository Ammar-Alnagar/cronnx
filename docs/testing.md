# Testing Documentation

## Overview

The Cronnx project includes a comprehensive test suite covering all major components and functionality. The test suite is designed to ensure reliability, correctness, and performance of the inference engine.

## Test Structure

### Unit Tests
- **Location**: Integrated within each module using `#[cfg(test)]`
- **Purpose**: Test individual functions and components in isolation
- **Coverage**: All core functions including error handling, preprocessing, and utility functions

### Integration Tests
- **Location**: `src/server/tests.rs`, `src/integration_tests.rs`
- **Purpose**: Test interactions between multiple components
- **Coverage**: HTTP API endpoints, model registry, batching system integration

### End-to-End Tests
- **Location**: `src/integration_tests.rs`
- **Purpose**: Test complete system flow from request to response
- **Coverage**: Full pipeline validation including error propagation

## Running Tests

### All Tests
```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run with specific test filter
cargo test health_check
```

### Specific Test Categories
```bash
# Run only unit tests
cargo test --lib

# Run specific module tests
cargo test error
cargo test preprocessing
cargo test batching
cargo test server
cargo test model

# Run integration tests only
cargo test --test integration_tests
```

## Test Coverage by Module

### Error Module (`src/error.rs`)
- **Tests**: 8 unit tests
- **Coverage**: 
  - Custom error type creation and conversion
  - Error message formatting
  - HTTP response generation
  - All error variant handling

### Preprocessing Module (`src/preprocessing/image.rs`)
- **Tests**: 9 unit tests
- **Coverage**:
  - Image loading and resizing
  - Tensor normalization using ImageNet stats
  - Shape validation and consistency
  - Error handling for invalid images
  - Different image formats and sizes

### Model Loading (`src/model/loader.rs`)
- **Tests**: 4 unit tests
- **Coverage**:
  - ORT initialization
  - File existence validation
  - Error handling for invalid models
  - Path type conversion

### Batching System (`src/batching/queue.rs`)
- **Tests**: 7 unit tests
- **Coverage**:
  - Batcher configuration
  - Job creation and transmission
  - Tensor concatenation logic
  - Batch size metrics
  - Async behavior validation

### Model Registry (`src/model/registry.rs`)
- **Tests**: 7 unit tests
- **Coverage**:
  - Model registration and lookup
  - Concurrent access patterns
  - Channel management
  - Registry cloning behavior

### Server Handlers (`src/server/handlers.rs`)
- **Tests**: 5 integration tests
- **Coverage**:
  - Health check endpoint
  - Prediction request handling
  - Error propagation
  - Invalid model handling
  - Base64 decoding validation

### End-to-End Integration (`src/integration_tests.rs`)
- **Tests**: 6 comprehensive tests
- **Coverage**:
  - Full system pipeline
  - Multi-model integration
  - Error propagation through system
  - Tensor processing pipeline
  - Config structure validation

### Observability (`src/observability_tests.rs`)
- **Tests**: 5 tests
- **Coverage**:
  - Metrics recording
  - Prometheus metrics collection
  - Tracing setup
  - Labeled metrics

## Test Quality Standards

### Assertion Coverage
- All tests include specific assertions
- Edge cases are thoroughly tested
- Error conditions are validated
- Success paths are verified

### Async Testing
- Proper timeouts for async operations
- Correct handling of futures
- Race condition testing
- Concurrent access validation

### Mocking Strategy
- Realistic mock components
- Proper channel communication testing
- Dependency isolation
- Behavior verification

## Performance Testing

### Load Testing
```bash
# Example load test setup
cargo test --release
# Then use external tools like:
# wrk, hey, or jmeter for performance testing
```

### Memory Usage
- Tests validate memory allocation patterns
- Tensor size validation
- Channel buffer management

## Continuous Integration

### Test Commands
```bash
# Basic test run
cargo test

# Format check
cargo fmt -- --check

# Lint check
cargo clippy -- -D warnings

# Build check
cargo build
cargo build --release
```

### Test Categories
- **Fast Unit Tests**: < 100ms execution time
- **Integration Tests**: < 1s execution time  
- **End-to-End Tests**: < 2s execution time

## Testing Best Practices

### Test Organization
- Clear test function names
- Setup/teardown patterns
- Input validation
- Output verification

### Error Testing
- All error paths covered
- Proper error message validation
- HTTP status code verification
- Error propagation testing

### Performance Considerations
- Minimal test execution time
- Efficient mock implementations
- Resource cleanup
- Parallel test execution support

## Adding New Tests

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_behavior() {
        // Test setup
        let input = create_test_input();
        
        // Execute function
        let result = function_to_test(input);
        
        // Verify output
        assert_eq!(result, expected_output);
    }
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_async_component() {
    // Async test setup
    let component = create_component();
    
    // Execute async function
    let result = component.async_function().await;
    
    // Verify result
    assert!(result.is_ok());
}
```

## Test Maintenance

### Regular Maintenance Tasks
- Update tests when changing APIs
- Add tests for new functionality
- Review and refactor complex tests
- Ensure test coverage remains high

### Coverage Verification
- All public functions have tests
- Error paths are covered
- Edge cases are tested
- Integration points are validated

## Test Environment

### Dependencies
- `tempfile`: For temporary file testing
- `tokio`: For async test execution
- Standard Rust testing framework

### Environment Variables
- Tests run without external dependencies
- Mock components for external services
- In-memory operations where possible

This comprehensive test suite ensures the reliability and correctness of the Cronnx inference engine across all functionality areas.