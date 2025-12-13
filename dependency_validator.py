import os
import subprocess
import sys
from typing import Tuple, List


class DependencyValidator:
    """Validates system and Python dependencies."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_ffmpeg(self) -> bool:
        """
        Check if FFmpeg is installed and accessible.
        
        Returns:
            bool: True if FFmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ FFmpeg is installed")
                return True
            else:
                self.errors.append("FFmpeg is installed but returned an error")
                return False
        except FileNotFoundError:
            self.errors.append(
                "FFmpeg is not installed or not in PATH.\n"
                "  Please install FFmpeg:\n"
                "  - Windows: Download from https://ffmpeg.org/download.html\n"
                "  - Linux: sudo apt-get install ffmpeg\n"
                "  - macOS: brew install ffmpeg"
            )
            return False
        except subprocess.TimeoutExpired:
            self.warnings.append("FFmpeg check timed out")
            return False
        except Exception as e:
            self.errors.append(f"Error checking FFmpeg: {str(e)}")
            return False

    def check_environment_variables(self) -> bool:
        """
        Check if required environment variables are set.
        
        Returns:
            bool: True if all required variables are set, False otherwise
        """
        required_vars = {
            'OPEN_AI_API_KEY': 'OpenAI API key for Whisper transcription',
            'GEMINI_API_KEY': 'Gemini API key for post-processing',
            'GEMINI_URL': 'Gemini API URL'
        }
        
        all_set = True
        for var, description in required_vars.items():
            if os.getenv(var):
                print(f"{var} is set")
            else:
                self.errors.append(
                    f"{var} is not set.\n"
                    f"  Purpose: {description}\n"
                    f"  Please add it to your .env file"
                )
                all_set = False
        
        return all_set

    def check_python_packages(self) -> bool:
        """
        Check if required Python packages are installed.
        
        Returns:
            bool: True if all packages are available, False otherwise
        """
        required_packages = {
            'torch': 'PyTorch',
            'soundfile': 'soundfile',
            'audio_separator': 'audio-separator',
            'openai': 'openai',
            'dotenv': 'python-dotenv',
            'requests': 'requests'
        }
        
        all_available = True
        for module_name, package_name in required_packages.items():
            try:
                __import__(module_name)
                print(f"✓ {package_name} is installed")
            except ImportError:
                self.errors.append(
                    f"Python package '{package_name}' is not installed.\n"
                    f"  Install with: pip install {package_name}"
                )
                all_available = False
        
        return all_available

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.
        
        Returns:
            Tuple[bool, List[str], List[str]]: (success, errors, warnings)
        """
        print("Validating dependencies...\n")
        
        ffmpeg_ok = self.check_ffmpeg()
        print()
        
        env_ok = self.check_environment_variables()
        print()
        
        packages_ok = self.check_python_packages()
        print()
        
        success = ffmpeg_ok and env_ok and packages_ok
        
        if success:
            print("All dependencies are satisfied!")
        else:
            print("Some dependencies are missing or not configured properly.")
        
        return success, self.errors, self.warnings

    def print_report(self):
        """Print a detailed report of validation results."""
        success, errors, warnings = self.validate_all()
        
        if warnings:
            print("\n⚠ WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if errors:
            print("\n✗ ERRORS:")
            for error in errors:
                print(f"  {error}\n")
        
        if not success:
            print("\nPlease fix the above issues before running the application.")
            sys.exit(1)


def validate_dependencies() -> bool:
    """
    Convenience function to validate all dependencies.
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    validator = DependencyValidator()
    validator.print_report()
    return True


if __name__ == "__main__":
    validate_dependencies()
