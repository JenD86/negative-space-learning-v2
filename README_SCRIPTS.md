# Negative Space Learning - Scripts Documentation

## Overview

The **Negative Space Learning** project is an AI training system that teaches language models to perform intelligent disk cleanup operations in containerized environments. The system operates by having AI models discover network environments, generate cleanup strategies, and execute them to free storage space across multiple Docker containers.

## Project Structure

```
Negative-Space-Learning-main/
├── scripts/                    # Main execution scripts
│   ├── main.py                # Core training pipeline
│   └── data_collector.py      # Automated data collection system
├── src/                       # Source code modules
│   ├── agent.py              # AI agent functions
│   ├── helper.py             # Utility functions
│   ├── prompt.py             # Prompt templates
│   ├── meta.py               # Prompt decorator system
│   ├── genner/               # AI model backends
│   ├── tool/                 # Docker and code tools
│   └── typing/               # Type definitions
├── config/                   # Configuration files
└── docker/                   # Docker compose setup
```

## Main Scripts

### 1. `scripts/main.py` - Core Training Pipeline

**Purpose**: Executes the complete negative space learning pipeline for a single iteration.

**Key Functions**:
- **Environment Discovery**: Generates Python code to scan network environments and discover containers
- **Strategy Generation**: Creates diverse cleanup strategies based on discovered environments  
- **Strategy Execution**: Implements and runs cleanup code to free disk space
- **Training Data Collection**: Saves all prompts, responses, and metrics for model training

**Workflow**:
```
1. Load Configuration → 2. Initialize AI Model → 3. Connect to Containers
         ↓
4. Environment Discovery Phase:
   - Generate code to scan network and containers
   - Execute code to gather environment information
   - Retry on failures with error feedback
         ↓
5. Strategy Generation Phase:
   - Generate 5 diverse cleanup strategies
   - Based on discovered environment data
   - Retry on parsing failures
         ↓
6. Strategy Execution Phase:
   - Select random strategy to execute
   - Generate Python code to implement strategy
   - Execute code and measure space freed
   - Retry on execution failures
         ↓
7. Save Results:
   - Training data (prompts + responses)
   - Execution metrics and space freed
   - Full run logs for analysis
```

**Key Features**:
- **Multi-Container Support**: Works across networked Docker containers
- **Error Recovery**: Automatic retry with error context for failed generations
- **Space Measurement**: Tracks disk space freed using multiple methods
- **Training Data**: Saves structured data for model fine-tuning

### 2. `scripts/data_collector.py` - Automated Data Collection

**Purpose**: Orchestrates large-scale automated data collection by running main.py repeatedly with different scenarios.

**Key Functions**:
- **Container Management**: Automatically restarts and manages Docker containers
- **Scenario Variation**: Creates diverse file cleanup scenarios across containers
- **Data Grouping**: Groups collected data by prompt types for training
- **Metrics Tracking**: Comprehensive tracking of cleanup efficiency and success rates

**Collection Scenarios**:
1. **Heavy Mix**: Large files across all categories (15.5MB)
2. **Medium Mix**: Balanced file sizes and counts (11.5MB) 
3. **Many Small**: Numerous small files (8.2MB)
4. **Large Sparse**: Few but very large files (19.4MB)
5. **Realistic Mixed**: Simulates real system conditions (16MB)

**Workflow**:
```
1. Initialize Containers → 2. Create File Scenario → 3. Run main.py
         ↓                        ↓                       ↓
4. Measure Results ← 5. Process Training Data ← 6. Group by Prompts
         ↓
7. Repeat until target data collected
         ↓
8. Generate Consolidated Training Dataset
```

**Key Features**:
- **Automated Scaling**: Collects thousands of training examples
- **Container Health**: Monitors and restarts containers as needed
- **Data Quality**: Tracks success rates and cleanup efficiency
- **Training Format**: Outputs data ready for model fine-tuning

## Source Code Architecture

### Core Modules

#### `src/agent.py` - AI Agent Interface
Provides high-level functions for interacting with AI models:
- `generate_special_environment_getter_code()`: Environment discovery
- `generate_strategy_list()`: Strategy generation  
- `generate_strategy_code()`: Strategy implementation
- `regenerate_*()`: Error recovery functions

#### `src/prompt.py` - Prompt Templates
Contains all prompt templates using the `@prompt` decorator:
- **System Prompt**: Defines the AI agent's role and constraints
- **Environment Discovery**: Templates for network scanning
- **Strategy Generation**: Templates for cleanup strategy creation
- **Strategy Execution**: Templates for code implementation
- **Error Recovery**: Templates for handling failures

#### `src/meta.py` - Prompt Decorator System
Implements the `@prompt` decorator that:
- Validates prompt templates at decoration time
- Formats prompts with runtime arguments
- Provides static analysis for common template errors
- Returns structured prompt objects

#### `src/helper.py` - Utility Functions
Core utility functions:
- `generate_readable_run_id()`: Creates unique run identifiers
- `get_formatted_repo_info()`: Git repository information
- `string_hash()`: Consistent string hashing
- `timeout()`: Execution timeout context manager

### Supporting Components

#### `src/genner/` - AI Model Backends
Abstracted AI model interfaces supporting:
- **Ollama**: Local model serving
- **OpenAI**: API-based models  
- **vLLM**: High-performance inference
- **PEFT**: Parameter-efficient fine-tuning

#### `src/tool/` - Execution Tools
- `docker.py`: Container management and code execution
- `code.py`: Python code validation

#### `src/typing/` - Type Definitions
- Configuration schemas
- Training data structures
- Message formats

## System Interactions

### Data Flow

```
Configuration → AI Model → Prompt Templates → Generated Code → Container Execution → Results
     ↓              ↓            ↓               ↓                ↓                ↓
Training Data ← Metrics ← Space Freed ← Execution Output ← Container State ← Code Validation
```

### Key Interactions

1. **main.py ↔ src/agent.py**: Main script calls agent functions for AI interactions
2. **agent.py ↔ src/prompt.py**: Agent uses prompt templates for model inputs  
3. **agent.py ↔ src/genner/**: Agent interfaces with AI models through genner abstraction
4. **main.py ↔ src/tool/docker.py**: Executes generated code in containers and measures results
5. **data_collector.py ↔ main.py**: Orchestrates multiple main.py runs with different scenarios

### Error Handling Flow

```
Code Generation → Validation → Execution → Error Detection → Regeneration with Context → Success
      ↑                                                              ↓
      └─────────────────────── Max Retries Exceeded ←──────────────┘
                                        ↓
                                   Failure Logged
```

## Configuration

### `config/config-container.toml`
Main configuration file specifying:
- **Model Settings**: Which AI model to use (Ollama, OpenAI, vLLM, PEFT)
- **Container IDs**: Target Docker containers for cleanup
- **Retry Limits**: Maximum attempts for each phase
- **Output Paths**: Where to save training data and results

### Environment Variables
- Container connection settings
- Model API keys (if using external models)
- Docker daemon configuration

## Usage Examples

### Running Single Iteration
```bash
# Use default config
python scripts/main.py

# Use custom config  
python scripts/main.py ./config/custom-config.toml
```

### Collecting Training Data
```bash
# Collect 20,000 training examples (default)
python scripts/data_collector.py

# Test run with 100 examples
python scripts/data_collector.py --test

# Collect 5000 examples with 3 completions per prompt
python scripts/data_collector.py --target 5000 --completions 3
```

## Output Data

### Training Data Structure
```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "raw_response": "Generated model response",
  "extract_result_if_ok": "Extracted code",
  "run_result_if_ok": "Execution output", 
  "this_code_space_change_kb": 1500,
  "success": true
}
```

### Metrics Tracking
- **Space Freed**: Actual disk space cleaned up
- **Efficiency**: Percentage of potential cleanup achieved  
- **Success Rates**: Code generation and execution success
- **Container Coverage**: Whether all containers were cleaned

## Key Design Principles

1. **Negative Space Learning**: Learning from the "negative space" of what should be cleaned up
2. **Multi-Container Awareness**: Training models to work across networked environments
3. **Real Execution**: Actual code execution rather than simulation
4. **Error-Driven Learning**: Using failures to improve model responses
5. **Diverse Scenarios**: Training on varied environmental conditions

## Dependencies

- **Docker**: Container orchestration and execution environment
- **AI Models**: Ollama, OpenAI API, or vLLM for code generation
- **Python Libraries**: `toml`, `docker`, `pydantic`, `loguru`, `result`

This system provides a comprehensive framework for training AI models to perform intelligent system administration tasks through practical, hands-on learning in containerized environments.
