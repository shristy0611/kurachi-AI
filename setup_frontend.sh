#!/bin/bash

echo "🚀 Setting up Kurachi AI Frontend & Backend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 16+ first."
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 16 ]; then
        print_error "Node.js version 16+ is required. Current version: $(node -v)"
        exit 1
    fi
    
    print_success "Node.js $(node -v) is installed"
}

# Check if npm is installed
check_npm() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm first."
        exit 1
    fi
    
    print_success "npm $(npm -v) is installed"
}

# Install frontend dependencies
install_frontend_deps() {
    print_status "Installing frontend dependencies..."
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        npm install
        if [ $? -eq 0 ]; then
            print_success "Frontend dependencies installed successfully"
        else
            print_error "Failed to install frontend dependencies"
            exit 1
        fi
    else
        print_warning "node_modules already exists, skipping installation"
    fi
    
    cd ..
}

# Build frontend
build_frontend() {
    print_status "Building React frontend..."
    
    cd frontend
    
    # Create build directory if it doesn't exist
    mkdir -p build
    
    # Build the React app
    npm run build
    if [ $? -eq 0 ]; then
        print_success "Frontend built successfully"
    else
        print_error "Failed to build frontend"
        exit 1
    fi
    
    cd ..
}

# Install backend dependencies
install_backend_deps() {
    print_status "Installing backend dependencies..."
    
    # Install FastAPI and uvicorn if not already installed
    pip install fastapi uvicorn python-multipart
    
    print_success "Backend dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p frontend/build
    mkdir -p logs
    
    print_success "Directories created"
}

# Start the backend server
start_backend() {
    print_status "Starting FastAPI backend server..."
    
    # Check if backend/main.py exists
    if [ ! -f "backend/main.py" ]; then
        print_error "backend/main.py not found"
        exit 1
    fi
    
    # Start the server
    cd backend
    python main.py &
    BACKEND_PID=$!
    cd ..
    
    # Wait a moment for the server to start
    sleep 3
    
    # Check if server is running
    if curl -s http://localhost:8000/api/health > /dev/null; then
        print_success "Backend server started successfully on http://localhost:8000"
        print_success "API documentation available at http://localhost:8000/api/docs"
    else
        print_error "Failed to start backend server"
        exit 1
    fi
}

# Main execution
main() {
    echo "🧠 Kurachi AI - Frontend & Backend Setup"
    echo "========================================"
    
    # Check prerequisites
    check_node
    check_npm
    
    # Create directories
    create_directories
    
    # Install dependencies
    install_frontend_deps
    install_backend_deps
    
    # Build frontend
    build_frontend
    
    # Start backend
    start_backend
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📱 Frontend: http://localhost:8000"
    echo "🔧 Backend API: http://localhost:8000/api"
    echo "📚 API Docs: http://localhost:8000/api/docs"
    echo ""
    echo "To stop the server, press Ctrl+C"
    echo ""
    
    # Keep the script running
    wait $BACKEND_PID
}

# Run main function
main
