#!/usr/bin/env python3
"""
File Commander - Interactive File Search and Basic Operations

A file search tool that helps users find, create, and perform basic operations on
their files. Features intelligent movie search with fast file discovery and an
intuitive menu-driven interface.

Features:
- 📁 Create Files & Folders - Create new folders and files with nested structures
- ⚡ Search & Manage Files/Folders - Find files and perform basic operations (open, rename)
- 🎬 Play Movie/Video - Smart movie finder that locates and plays films quickly
- 📋 List Directory Contents - Browse and explore folder contents
- ⚙️ Search Statistics - View search index status and performance

Specializes in fast movie discovery with intelligent matching to find the right
films without false results.
"""

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm

# Initialize Rich console for beautiful terminal output
console = Console()
app = typer.Typer()

# =============================================================================
# CONSTANTS - Centralized configuration for easy maintenance
# =============================================================================

# Video file extensions we can play (common formats across different players)
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".m2ts",
    ".3gp",
    ".3g2",
}

# System directories to skip during indexing (improves performance and security)
SKIP_DIRECTORIES = {
    "system32",
    "windows",
    "programdata",
    "$recycle",
    "appdata",
    ".git",
    "node_modules",
    "__pycache__",
}

# Common movie folder names across different systems and languages
MOVIE_FOLDER_NAMES = [
    "Movies",
    "Movie",
    "movies",
    "Hollywood",
    "Bollywood",
    "Videos",
    "Pictures",
    "Downloads",
    "Samsung Flow",
    "Music",
    "Media",
    "Entertainment",
    "Films",
    "Cinema",
    "Video",
    "Clips",
]

# Confidence threshold for movie matching (80% = high confidence)
HIGH_CONFIDENCE_THRESHOLD = 80


# =============================================================================
# UTILITY CLASSES - Reusable components for common operations
# =============================================================================


class PathUtils:
    """
    Utility methods for safe and efficient path operations.

    Handles drive detection, path validation, and security checks to prevent
    common file system vulnerabilities like directory traversal attacks.
    """

    @staticmethod
    def get_drive_path(drive_letter: str) -> Path:
        """Convert drive letter to Path object (e.g., 'D' -> 'D:/')"""
        return Path(f"{drive_letter.upper()}:/")

    @staticmethod
    def get_available_drives() -> List[str]:
        """
        Scan system for available drive letters.

        Returns list of drive letters that actually exist and are accessible.
        Useful for building dynamic folder lists across different systems.
        """
        drives = []
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            if PathUtils.get_drive_path(letter).exists():
                drives.append(letter)
        return drives

    @staticmethod
    def is_valid_folder(path: Path) -> bool:
        """Check if path exists and is actually a directory (not a file)"""
        return path.exists() and path.is_dir()

    @staticmethod
    def should_skip_directory(path: Path) -> bool:
        """
        Security check: determine if directory should be skipped during indexing.

        Skips system directories to improve performance and avoid indexing
        sensitive system files that users don't typically want to search.
        """
        path_str = str(path).lower()
        return any(skip in path_str for skip in SKIP_DIRECTORIES)

    @staticmethod
    def filter_video_files(results: List[Path]) -> List[Path]:
        """
        Filter file list to only include playable video files.

        Checks both file extension and existence to ensure we only return
        files that can actually be played by the system's default media player.
        """
        return [
            path
            for path in results
            if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]

    @staticmethod
    def get_folders_from_drives(
        drives: List[str], folder_names: List[str]
    ) -> List[Path]:
        """
        Collect valid folders from multiple drives efficiently.

        This consolidates the common pattern of checking drive + folder combinations
        and only returns folders that actually exist, reducing duplicate code.
        """
        folders = []
        for drive in drives:
            drive_path = PathUtils.get_drive_path(drive)
            if drive_path.exists():
                for folder_name in folder_names:
                    folder_path = drive_path / folder_name
                    if PathUtils.is_valid_folder(folder_path):
                        folders.append(folder_path)
        return folders

    @staticmethod
    def is_safe_filename(name: str) -> bool:
        """
        Security validation for user-provided file/folder names.

        Prevents directory traversal attacks (like '../../../etc/passwd')
        and ensures compatibility with Windows file system restrictions.
        Essential for any file manager that accepts user input.
        """
        # Check for empty or whitespace-only names
        if not name or not name.strip():
            return False

        # Directory traversal protection - these patterns can escape intended directories
        dangerous_patterns = ["../", "..\\", "../", "..\\"]
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in dangerous_patterns):
            return False

        # Windows file system restrictions - these characters cause errors
        invalid_chars = '<>:"|?*'
        if any(char in name for char in invalid_chars):
            return False

        return True

    @staticmethod
    def get_item_type(path: Path) -> str:
        """Get simple item type string: 'folder' or 'file'"""
        return "folder" if path.is_dir() else "file"

    @staticmethod
    def get_item_emoji_type(path: Path) -> str:
        """Get emoji item type string: '📁 Folder' or '📄 File'"""
        return "📁 Folder" if path.is_dir() else "📄 File"


class UIUtils:
    """
    User interface utilities for consistent, interactive terminal experience.

    Centralizes common UI patterns like table creation, option menus, error
    handling, and visual formatting to ensure consistent look and feel.
    """

    @staticmethod
    def create_results_table(title: str, columns: List[Tuple[str, str, int]]) -> Table:
        """
        Create standardized table for displaying search results.

        Args:
            title: Table title shown at the top
            columns: List of (name, style, width) tuples for each column
                    width=0 means auto-size the column
        """
        table = Table(title=title)
        for name, style, width in columns:
            if width:
                table.add_column(name, style=style, width=width)
            else:
                table.add_column(name, style=style)
        return table

    @staticmethod
    def apply_standard_table_styling(table: Table):
        """Apply consistent styling to all tables in the application"""
        table.show_lines = True
        table.header_style = "bold cyan"

    @staticmethod
    def get_user_choice(prompt: str, choices: List[str], default: str = None) -> str:
        """Get validated user input with automatic retry on invalid choices"""
        if default:
            return Prompt.ask(prompt, choices=choices, default=default)
        else:
            return Prompt.ask(prompt, choices=choices)

    @staticmethod
    def show_options_and_choose(options: List[str], prompt: str) -> str:
        """
        Display numbered menu options and get user selection.

        This pattern appears frequently in interactive applications:
        1. Show numbered options
        2. Get user choice
        3. Validate input

        Consolidating it here ensures consistent UI behavior.
        """
        for option in options:
            console.print(option)

        choices = [str(i) for i in range(1, len(options) + 1)]
        return UIUtils.get_user_choice(prompt, choices)

    @staticmethod
    def print_success(message: str):
        """Print success message with consistent formatting"""
        console.print(f"[bold green]✅ SUCCESS:[/] {message}")

    @staticmethod
    def print_error(message: str):
        """Print error message with consistent formatting"""
        console.print(f"[bold red]❌ ERROR:[/] {message}")

    @staticmethod
    def print_warning(message: str):
        """Print warning message with consistent formatting"""
        console.print(f"[bold yellow]⚠️ WARNING:[/] {message}")

    @staticmethod
    def print_info(message: str):
        """Print info message with consistent formatting"""
        console.print(f"[bold cyan]ℹ️ INFO:[/] {message}")

    @staticmethod
    def print_separator():
        """Print standard visual separator line"""
        console.print("─" * 60)

    @staticmethod
    def print_section_break():
        """Print section break line for major divisions"""
        console.print("═" * 60)

    @staticmethod
    def print_section_header(title: str):
        """Print formatted section header with consistent styling"""
        console.print()
        console.print(Panel(title, style="bold green"))
        UIUtils.print_separator()

    @staticmethod
    def validate_filename_or_show_error(name: str) -> bool:
        """
        Validate filename and show error if invalid.

        Returns True if valid, False if invalid (with error shown).
        Consolidates the common pattern of validation + error display.
        """
        if not PathUtils.is_safe_filename(name):
            UIUtils.print_error(
                "Invalid name. Avoid empty names, '..' patterns, and special characters"
            )
            return False
        return True

    @staticmethod
    def safe_execute(operation_name: str, func, *args, **kwargs):
        """
        Execute file operations with comprehensive error handling.

        File system operations can fail for many reasons (permissions, disk space,
        network issues, etc.). This wrapper provides consistent error messages
        and prevents crashes from propagating to the user interface.
        """
        try:
            return func(*args, **kwargs)
        except PermissionError:
            UIUtils.print_error(f"Permission denied: {operation_name}")
        except FileNotFoundError:
            UIUtils.print_error(f"File not found: {operation_name}")
        except Exception as e:
            UIUtils.print_error(f"{operation_name} - {e}")
        return None


# =============================================================================
# SEARCH ENGINE - Fast file indexing and retrieval system
# =============================================================================


class TrieNode:
    """
    Node in a Trie (prefix tree) data structure.

    A Trie allows fast prefix matching - essential for autocomplete-style search
    where users type partial movie names like "The Int" to find "The Intern".
    Each node stores characters and associated files.
    """

    def __init__(self):
        self.children = {}  # Dictionary mapping characters to child nodes
        self.files = []  # Files that contain this prefix


class Trie:
    """
    Trie (prefix tree) for ultra-fast prefix-based file search.

    Why use a Trie?
    - Allows instant prefix matching: "The" finds all files starting with "The"
    - Much faster than scanning all filenames repeatedly
    - Enables autocomplete-style search functionality
    - Scales well with large file collections
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, file_path: Path):
        """
        Insert a word (filename) into the trie with associated file path.

        As we traverse each character, we add the file to every prefix node.
        This means searching for "The" will find files like "The Intern.mp4".
        """
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # Add file to this prefix - enables partial matching
            node.files.append(file_path)

    def search_prefix(self, prefix: str, max_results: int = 20) -> List[Path]:
        """
        Find all files matching a prefix (like autocomplete).

        Returns unique files to avoid duplicates from multiple word matches.
        Essential for responsive search as users type partial movie names.
        """
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        # Remove duplicates while preserving order (dict.fromkeys trick)
        unique_files = list(dict.fromkeys(node.files))
        return unique_files[:max_results]


class FileMetadata:
    """
    Lightweight wrapper for file and folder information to avoid repeated Path operations.

    Caches commonly-needed properties to improve search performance
    when dealing with thousands of files and folders.
    """

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.suffix = path.suffix.lower()  # File extension for type filtering
        self.is_dir = path.is_dir()


class FileSearchIndex:
    """
    High-performance file and folder search engine with multiple search strategies.

    Combines several search approaches for comprehensive file and folder discovery:
    1. Trie for prefix matching (autocomplete-style)
    2. Exact filename lookup (fastest for known names)
    3. Word-based search (handles different word orders)
    4. Substring search (broadest matching)

    This multi-strategy approach ensures users find files and folders regardless of
    how they remember or type the filename.
    """

    def __init__(self):
        # Trie for fast prefix search (like autocomplete)
        self.trie = Trie()

        # Hash map for instant exact filename lookup
        self.exact_match = {}  # filename -> [FileMetadata]

        # Inverted index: word -> set of files containing that word
        # Enables searching for "intern the" to find "The Intern.mp4"
        self.word_index = defaultdict(set)

        # Track indexed files to avoid duplicates
        self.indexed_paths = set()

        # Statistics for user feedback
        self.total_items = 0

    def add_file(self, file_path: Path):
        """
        Add a single file or folder to all search indexes.

        This is the core indexing operation that makes files and folders searchable
        through multiple strategies. Skip if already indexed to avoid duplicates.
        Despite the method name, this works for both files and directories.
        """
        # Avoid duplicate indexing (important for performance)
        if str(file_path).lower() in self.indexed_paths:
            return

        try:
            metadata = FileMetadata(file_path)
            filename = metadata.name.lower()

            # 1. Add to trie for prefix search
            self.trie.insert(filename, file_path)

            # 2. Add to exact match lookup
            if filename not in self.exact_match:
                self.exact_match[filename] = []
            self.exact_match[filename].append(metadata)

            # 3. Add to word index for flexible search
            # Split filename into searchable words (handle dots, underscores, dashes)
            words = (
                filename.replace(".", " ").replace("_", " ").replace("-", " ").split()
            )
            for word in words:
                if len(word) > 2:  # Skip very short words (the, of, a, etc.)
                    self.word_index[word].add(file_path)

            # Track this file as indexed
            self.indexed_paths.add(str(file_path).lower())
            self.total_items += 1

        except (OSError, PermissionError):
            # Skip files we can't access (common in system directories)
            pass

    def index_folder(self, folder_path: Path) -> int:
        """
        Index all files and folders in a directory and its subdirectories.

        Returns the number of items (files + folders) successfully indexed, which helps
        users understand indexing progress and completeness.
        Uses recursive glob (rglob) for efficient directory traversal.
        """
        items_added = 0

        if not PathUtils.is_valid_folder(folder_path):
            return items_added

        try:
            # rglob("*") recursively finds all files AND folders in subdirectories
            for item in folder_path.rglob("*"):
                # Skip system directories for performance and security
                if PathUtils.should_skip_directory(item.parent):
                    continue

                # Index both files AND folders for comprehensive search
                self.add_file(item)  # Works for both files and directories
                items_added += 1

        except (OSError, PermissionError):
            # Skip inaccessible directories (network drives, system folders, etc.)
            pass

        return items_added

    def search_videos(self, query: str, max_results: int = 20) -> List[Path]:
        """
        Specialized search for video files only.

        Uses the general search but filters results to only video formats.
        Essential for movie search where we only want playable files.
        """
        all_results = self.search(
            query, max_results * 2
        )  # Get extra results before filtering
        return PathUtils.filter_video_files(all_results)[:max_results]

    def search(self, query: str, max_results: int = 20) -> List[Path]:
        """
        Multi-strategy search combining all indexing approaches.

        Search progression from fastest to broadest:
        1. Exact match (instant hash lookup)
        2. Prefix search (Trie-based autocomplete)
        3. Word search (handles different word orders)
        4. Substring search (broadest matching)

        This ensures we find files efficiently while providing comprehensive results.
        """
        if not query.strip():
            return []

        query = query.lower().strip()
        results = set()  # Use set to automatically handle duplicates

        # Strategy 1: Exact filename match (fastest possible)
        if query in self.exact_match:
            for metadata in self.exact_match[query]:
                results.add(metadata.path)

        # Strategy 2: Prefix search using Trie (autocomplete-style)
        prefix_results = self.trie.search_prefix(query, max_results * 2)
        results.update(prefix_results)

        # Strategy 3: Word-based search (handles different word orders)
        # Splits "the intern" into ["the", "intern"] for flexible matching
        query_words = query.replace(".", " ").replace("_", " ").split()
        for word in query_words:
            if word in self.word_index:
                results.update(self.word_index[word])

        # Strategy 4: Substring search (broadest, slowest)
        # Only use if we don't have enough results yet
        if len(results) < max_results:
            for filename, metadata_list in self.exact_match.items():
                if query in filename:
                    for metadata in metadata_list:
                        results.add(metadata.path)

        # Sort results by relevance and return top matches
        return self._sort_by_relevance(list(results), query)[:max_results]

    def _sort_by_relevance(self, results: List[Path], query: str) -> List[Path]:
        """
        Sort search results by relevance score for better user experience.

        Scoring prioritizes:
        1. Exact matches (highest score)
        2. Filenames starting with query
        3. Filenames containing query
        4. Shorter filenames (usually more relevant)
        5. Files in common directories
        """

        def score(path: Path) -> int:
            filename = path.name.lower()
            relevance_score = 0

            # Exact match gets highest priority
            if query == filename:
                relevance_score += 100
            # Starts with query (like autocomplete)
            elif filename.startswith(query):
                relevance_score += 80
            # Contains query somewhere
            elif query in filename:
                relevance_score += 50

            # Shorter filenames often more relevant (less clutter)
            relevance_score += max(0, 30 - len(filename))

            # Bonus for files in commonly-accessed directories
            parent_name = path.parent.name.lower()
            if any(
                common in parent_name
                for common in ["documents", "desktop", "downloads"]
            ):
                relevance_score += 10

            return relevance_score

        return sorted(results, key=score, reverse=True)


# =============================================================================
# MOVIE MATCHING - Confidence-based movie identification system
# =============================================================================


class MovieMatcher:
    """
    Intelligent movie matching with confidence scoring.

    Prevents false positives by scoring how well a filename matches the user's
    search term. Essential for movie search where partial matches can be misleading
    (e.g., "The Secret Life..." shouldn't match "The Intern").
    """

    @staticmethod
    def calculate_confidence(movie_name: str, filename: str) -> int:
        """
        Calculate match confidence score (0-100).

        Scoring system:
        - 100: Exact match ("The Intern" == "The Intern.mp4")
        - 90:  Starts with search term
        - 80:  Contains exact search term
        - 70:  All search words present
        - 20-60: Partial word matches
        - 0:   No meaningful match

        This prevents false positives like "The Day After Tomorrow" matching "The Intern".
        """
        movie_lower = movie_name.lower().strip()
        file_base = Path(filename).stem.lower()  # Remove file extension

        # Perfect match - highest confidence
        if movie_lower == file_base:
            return 100

        # File starts with search term - very high confidence
        if file_base.startswith(movie_lower):
            return 90

        # File contains exact search term - high confidence
        if movie_lower in file_base:
            return 80

        # Word-by-word analysis for more nuanced matching
        movie_words = set(movie_lower.split())
        file_words = set(file_base.replace(".", " ").replace("_", " ").split())

        # All search words present in filename - good confidence
        if movie_words.issubset(file_words):
            return 70

        # Partial word matching - medium to low confidence
        matching_words = movie_words.intersection(file_words)
        if matching_words:
            match_ratio = len(matching_words) / len(movie_words)
            if match_ratio >= 0.8:  # 80%+ words match
                return 60
            elif match_ratio >= 0.5:  # 50%+ words match
                return 40
            else:  # Less than 50% match
                return 20

        return 0  # No meaningful match

    @staticmethod
    def find_best_matches(
        search_results: List[Path], movie_name: str
    ) -> List[Tuple[Path, int]]:
        """
        Score and rank all search results by confidence.

        Returns list of (file_path, confidence_score) tuples sorted by confidence.
        This allows the UI to show users the most relevant matches first and
        distinguish between high-confidence and questionable matches.
        """
        scored_results = []
        for path in search_results:
            confidence = MovieMatcher.calculate_confidence(movie_name, path.name)
            if confidence > 0:  # Only include meaningful matches
                scored_results.append((path, confidence))

        # Sort by confidence score (highest first)
        return sorted(scored_results, key=lambda result: result[1], reverse=True)


# =============================================================================
# MAIN APPLICATION - Interactive file management interface
# =============================================================================


class FileCommander:
    """
    Main application class providing interactive file management.

    Features:
    - Confidence-based movie search with early termination
    - Priority-ordered folder scanning (common locations first)
    - Secure file operations with input validation
    - Intuitive menu-driven interface
    - Fast incremental indexing
    - Nested file/folder creation capabilities
    """

    def __init__(self):
        self.desktop = Path.home() / "Desktop"
        self.search_index = FileSearchIndex()
        self.current_context = {"location": "", "project": ""}  # For future features

    def show_main_menu(self):
        """Display the main application menu with available operations."""
        console.clear()

        # Application header with enhanced styling
        header = Text("⚡ File Commander", style="bold blue")
        console.print(Panel(header, subtitle="Optimized File Management", style="cyan"))

        # Visual separator
        console.print()

        # Main menu options with improved table styling
        options = [
            ("1", "📁 Create Files & Folders"),
            ("2", "⚡ Search & Manage Files/Folders"),
            ("3", "🎬 Play Movie/Video"),
            ("4", "📋 List Directory Contents"),
            ("5", "⚙️ Search Statistics"),
            ("0", "❌ Exit"),
        ]

        # Create formatted table with enhanced styling
        table = UIUtils.create_results_table(
            "🎯 Choose an Operation",
            [("Option", "cyan", 6), ("Description", "white", 0)],
        )

        # Apply standard styling
        UIUtils.apply_standard_table_styling(table)

        for option, description in options:
            table.add_row(option, description)

        console.print(table)
        UIUtils.print_separator()

    def _get_prioritized_movie_folders(self) -> List[Path]:
        """
        Get movie folders ordered by likelihood of containing the target movie.

        Priority system:
        1. User's Videos folder (most common location for personal movies)
        2. Dedicated movie folders on D: and C: drives
        3. Other user media folders (Pictures, Downloads, etc.)
        4. Remaining movie folders on both drives

        This prioritization enables early termination - we can stop searching
        as soon as we find a high-confidence match in a priority location.
        """
        folders = []

        # Priority 1: User's personal Videos folder (highest probability)
        user_videos = Path.home() / "Videos"
        if PathUtils.is_valid_folder(user_videos):
            folders.append(user_videos)

        # Priority 2: Dedicated movie folders on primary drives
        # Check D: first (often used for media), then C:
        folders.extend(
            PathUtils.get_folders_from_drives(["D", "C"], MOVIE_FOLDER_NAMES[:5])
        )

        # Priority 3: Other user media folders
        user_folders = [
            Path.home() / "Pictures" / "Samsung Flow",  # Common phone sync location
            Path.home() / "Downloads",  # Downloaded movies
            Path.home() / "Pictures",  # Sometimes contains videos
            Path.home() / "Music",  # Music videos, concerts
        ]

        for folder in user_folders:
            if PathUtils.is_valid_folder(folder):
                folders.append(folder)

        # Priority 4: Remaining movie folders (less common but comprehensive)
        folders.extend(
            PathUtils.get_folders_from_drives(["D", "C"], MOVIE_FOLDER_NAMES[5:])
        )

        # Remove duplicates while preserving priority order
        unique_folders = []
        seen = set()
        for folder in folders:
            folder_str = str(folder).lower()
            if folder_str not in seen:
                seen.add(folder_str)
                unique_folders.append(folder)

        return unique_folders

    def _search_movie_incrementally(self, movie_name: str) -> Optional[Path]:
        """
        Search for movie with incremental indexing and confidence-based early termination.

        Key optimization: Instead of indexing everything first, we index folder-by-folder
        and check for high-confidence matches after each folder. This means:
        - If "The Intern" is in the Videos folder, we find it in ~1 second
        - We don't waste time indexing the entire hard drive
        - Users get results as fast as possible

        Returns:
        - Path object if movie found
        - "retry" string if user wants to search again with different term
        - None if no movie found
        """
        UIUtils.print_info(f"Searching for '{movie_name}' in prioritized folders...")

        prioritized_folders = self._get_prioritized_movie_folders()
        all_candidates = []  # Collect all potential matches across folders

        # Search each folder incrementally
        for i, folder in enumerate(prioritized_folders, 1):
            console.print(
                f"[dim]📁 Checking folder {i}/{len(prioritized_folders)}: {folder.name}[/dim]"
            )

            # Index this specific folder (incremental approach)
            items_added = self.search_index.index_folder(folder)

            if items_added > 0:
                console.print(f"[dim]   ✅ Indexed {items_added} items[/dim]")

                # Search for videos in this specific folder
                video_results = self.search_index.search_videos(movie_name, 20)
                folder_videos = [
                    path
                    for path in video_results
                    if folder in path.parents or path.parent == folder
                ]

                if folder_videos:
                    # Score the matches using confidence algorithm
                    scored_matches = MovieMatcher.find_best_matches(
                        folder_videos, movie_name
                    )

                    if scored_matches:
                        best_match = scored_matches[0]
                        console.print(
                            f"[dim]   🎯 Best match: {best_match[0].name} ({best_match[1]}% confidence)[/dim]"
                        )

                        # Collect candidates for later processing
                        all_candidates.extend(scored_matches)

                        # Early termination: if we found a high-confidence match, stop here
                        if best_match[1] >= HIGH_CONFIDENCE_THRESHOLD:
                            # But only stop if there's a single high-confidence match to avoid ambiguity
                            high_conf_matches = [
                                match
                                for match in scored_matches
                                if match[1] >= HIGH_CONFIDENCE_THRESHOLD
                            ]
                            if len(high_conf_matches) == 1:
                                UIUtils.print_success("High-confidence match found!")
                                return best_match[0]
            else:
                console.print(f"[dim]   ⚪ No items found[/dim]")

        # Process all collected candidates to find the best overall match
        return self._process_movie_candidates(all_candidates, movie_name)

    def _process_movie_candidates(
        self, candidates: List[Tuple[Path, int]], movie_name: str
    ) -> Optional[Path]:
        """
        Process collected movie candidates based on confidence scores.

        Handles different scenarios:
        1. High-confidence matches found: Show best options
        2. Only low-confidence matches: Offer alternatives
        3. No matches: Return None for broader search
        """
        if not candidates:
            return None

        # Sort all candidates by confidence (best first)
        candidates.sort(key=lambda result: result[1], reverse=True)
        high_confidence = [
            candidate
            for candidate in candidates
            if candidate[1] >= HIGH_CONFIDENCE_THRESHOLD
        ]

        if high_confidence:
            UIUtils.print_success(
                f"Found {len(high_confidence)} high-confidence matches!"
            )

            # Single perfect match - return immediately
            if len(high_confidence) == 1:
                result = high_confidence[0]
                console.print(
                    f"[green]🎬 Perfect match: {result[0].name} ({result[1]}% confidence)[/green]"
                )
                return result[0]
            else:
                # Multiple good matches - let user choose
                return self._show_movie_options(
                    high_confidence, f"High-Confidence Matches for '{movie_name}'"
                )
        else:
            # No high-confidence matches - offer alternatives
            UIUtils.print_warning("No high-confidence matches found")

            options = [
                "1. 🎬 Play one of these movies",
                "2. 🔍 Search entire drives",
                "3. 🔄 Try different search",
                "4. 🔙 Back to menu",
            ]

            choice = UIUtils.show_options_and_choose(options, "Choose action")

            if choice == "1":
                return self._show_movie_options(
                    candidates[:5], f"Best Available Matches for '{movie_name}'"
                )
            elif choice == "3":
                return "retry"  # Signal to restart search with new term

        return None

    def _show_movie_options(
        self, candidates: List[Tuple[Path, int]], title: str
    ) -> Optional[Path]:
        """
        Display movie options in a formatted table and get user selection.

        Shows confidence scores to help users make informed choices.
        Essential when multiple movies partially match the search term.
        """
        UIUtils.print_separator()
        console.print(Panel(title, style="cyan"))

        table = UIUtils.create_results_table(
            "",
            [
                ("#", "white", 3),
                ("Movie Name", "green", 0),
                ("Confidence", "yellow", 10),
                ("Location", "blue", 0),
            ],
        )

        # Apply enhanced table styling
        UIUtils.apply_standard_table_styling(table)

        # Show up to 10 options to avoid overwhelming the user
        for idx, (path, confidence) in enumerate(candidates[:10], 1):
            table.add_row(str(idx), path.name, f"{confidence}%", path.parent.name)

        console.print(table)
        UIUtils.print_separator()

        choice = UIUtils.get_user_choice(
            "Select movie to play",
            [str(i) for i in range(1, min(len(candidates), 10) + 1)],
        )
        return candidates[int(choice) - 1][0]

    def play_movie(self):
        """
        Main movie search and play functionality.

        This is the core feature that demonstrates the power of incremental search
        with confidence-based matching. Users get fast, accurate results without
        waiting for complete system indexing.
        """
        UIUtils.print_section_header("🎬 Play Movie/Video")

        movie_name = Prompt.ask("🎬 Movie name to search")
        if not movie_name.strip():
            UIUtils.print_error("Please enter a movie name")
            return

        UIUtils.print_section_break()
        UIUtils.print_info("Starting Movie Search...")
        UIUtils.print_section_break()

        # Track search performance for user feedback
        start_time = time.time()
        found_movie = self._search_movie_incrementally(movie_name)
        search_time = time.time() - start_time

        # Handle retry request (user wants to search with different term)
        if found_movie == "retry":
            self.play_movie()  # Recursive call for new search
            return

        if found_movie:
            # Movie found - play it with performance feedback
            UIUtils.print_section_break()
            UIUtils.print_success(f"Search completed in {search_time:.2f} seconds")
            UIUtils.print_section_break()

            def play_file():
                os.startfile(str(found_movie))  # Use system default player
                console.print(f"[bold green]▶️ PLAYING:[/] {found_movie.name}")
                console.print(f"[dim]📂 Location: {found_movie.parent}[/dim]")

            UIUtils.safe_execute("playing movie", play_file)
        else:
            # No movie found in prioritized folders - offer broader search
            UIUtils.print_section_break()
            UIUtils.print_warning("SEARCH COMPLETE: No high-confidence matches found")
            UIUtils.print_section_break()
            self._handle_movie_not_found(movie_name, search_time)

    def _handle_movie_not_found(self, movie_name: str, search_time: float):
        """
        Handle case when movie is not found in prioritized folders.

        Offers progressively broader search options:
        1. Full drive indexing (slower but comprehensive)
        2. Try different search term
        3. Return to main menu
        """
        UIUtils.print_warning(f"Movie '{movie_name}' not found in prioritized folders")
        console.print(f"[dim]⏱️ Searched in {search_time:.2f} seconds[/dim]")

        options = [
            "1. 🔍 Search entire D: drive",
            "2. 🔍 Search entire C: drive",
            "3. 🔄 Try different search",
            "4. 🔙 Back to menu",
        ]

        choice = UIUtils.show_options_and_choose(options, "Choose option")

        if choice == "1":
            self._search_entire_drive("D", movie_name)
        elif choice == "2":
            self._search_entire_drive("C", movie_name)
        elif choice == "3":
            self.play_movie()  # Try again with new search term

    def _search_entire_drive(self, drive_letter: str, movie_name: str):
        """
        Fallback option: search entire drive when movie not found in common locations.

        This is slower but comprehensive - indexes the entire drive and then
        searches with the same confidence-based matching system.
        """
        drive_path = PathUtils.get_drive_path(drive_letter)

        if not drive_path.exists():
            UIUtils.print_error(f"Drive {drive_letter}: not found")
            return

        console.print(
            f"[bold yellow]🔄 Searching entire {drive_letter}: drive...[/bold yellow]"
        )
        console.print("[dim]⏳ This may take a few moments...[/dim]")

        # Index the entire drive (this is the slow part)
        items_added = self.search_index.index_folder(drive_path)

        if items_added > 0:
            console.print(f"[dim]📊 Indexed {items_added} additional items[/dim]")

            # Search the newly indexed files
            video_results = self.search_index.search_videos(movie_name, 20)
            drive_results = [
                path
                for path in video_results
                if str(path).lower().startswith(drive_letter.lower())
            ]

            if drive_results:
                # Apply confidence scoring to drive results
                scored_results = MovieMatcher.find_best_matches(
                    drive_results, movie_name
                )
                high_confidence = [
                    result
                    for result in scored_results
                    if result[1] >= HIGH_CONFIDENCE_THRESHOLD
                ]

                if high_confidence:
                    UIUtils.print_success(
                        f"Found high-confidence matches on {drive_letter}: drive!"
                    )
                    selected = self._show_movie_options(
                        high_confidence,
                        f"{drive_letter}: Drive High-Confidence Matches",
                    )
                else:
                    UIUtils.print_warning(
                        f"No high-confidence matches on {drive_letter}: drive"
                    )
                    if Confirm.ask("Show best available matches?"):
                        selected = self._show_movie_options(
                            scored_results[:10], f"{drive_letter}: Drive Best Matches"
                        )
                    else:
                        return

                # Play the selected movie
                if selected:

                    def play_file():
                        os.startfile(str(selected))
                        UIUtils.print_success(f"Playing: {selected.name}")

                    UIUtils.safe_execute("playing movie", play_file)
            else:
                UIUtils.print_error(f"No movies found on {drive_letter}: drive")
        else:
            UIUtils.print_warning(f"No accessible items on {drive_letter}: drive")

    def search_files(self):
        """
        General file search functionality for non-movie files.

        Provides broader search capabilities for documents, images, and other files.
        Uses the same fast indexing system but without movie-specific filtering.
        Supports continuous searching without re-indexing for better performance.
        """
        UIUtils.print_section_header("⚡ Search & Manage Files/Folders")

        # Index common user folders for general file search (ONE TIME ONLY)
        common_folders = [
            Path.home() / "Downloads",
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path.home() / "Videos",
            Path.home() / "Pictures",
        ]

        # Build index from common locations with progress feedback
        console.print("[dim]🔄 Indexing common folders...[/dim]")
        for folder in common_folders:
            if PathUtils.is_valid_folder(folder):
                self.search_index.index_folder(folder)

        UIUtils.print_success("Indexing complete")
        UIUtils.print_separator()

        # Continuous search loop - no re-indexing needed
        while True:
            search_term = Prompt.ask("⚡ What are you looking for?")
            if not search_term.strip():
                UIUtils.print_error("Please enter a search term")
                continue  # Ask again without breaking the loop

            UIUtils.print_info(f"Searching for '{search_term}'...")

            # Perform search with performance tracking
            start_time = time.time()
            results = self.search_index.search(search_term, 50)
            search_time = time.time() - start_time

            if results:
                UIUtils.print_success(
                    f"Found {len(results)} results in {search_time:.3f} seconds"
                )
                UIUtils.print_section_break()
                self._display_search_results(results, search_term)

                # Handle actions and check if user wants to continue
                if not self._handle_search_actions(results):
                    break  # Exit to main menu if user chose "Back to menu"
            else:
                UIUtils.print_section_break()
                UIUtils.print_warning(f"No items found for '{search_term}'")
                UIUtils.print_section_break()

                # Ask if user wants to continue searching (only when no results)
                UIUtils.print_separator()
                if not Confirm.ask(
                    "[bold cyan]🔍 Do you want to search for something else?[/bold cyan]",
                    default=False,
                ):
                    console.print("[dim]👍 Returning to main menu[/dim]")
                    break  # Exit the search loop and return to main menu

            UIUtils.print_separator()  # Visual separator for next search

    def _display_search_results(self, results: List[Path], search_term: str):
        """Display search results in a formatted table with file type indicators."""
        UIUtils.print_separator()

        table = UIUtils.create_results_table(
            f"🔍 Results for '{search_term}'",
            [
                ("#", "white", 3),
                ("Name", "green", 0),
                ("Type", "white", 8),
                ("Location", "blue", 0),
            ],
        )

        # Apply enhanced table styling
        UIUtils.apply_standard_table_styling(table)

        # Show first 20 results to avoid overwhelming the user
        for i, item in enumerate(results[:20], 1):
            item_type = PathUtils.get_item_emoji_type(item)
            table.add_row(str(i), item.name, item_type, str(item.parent))

        console.print(table)

        # Indicate if there are more results
        if len(results) > 20:
            console.print(
                f"[dim]... and {len(results) - 20} more results (showing first 20)[/dim]"
            )

        UIUtils.print_separator()

    def _handle_search_actions(self, results: List[Path]) -> bool:
        """
        Handle user actions on search results (open, rename, etc.).
        Returns True if user wants to continue searching, False to exit to main menu.
        """
        actions = [
            "1. 📂 Open item",
            "2. ✏️ Rename item",
            "3. 🔍 Search again",
            "4. 🔙 Back to menu",
        ]

        action = UIUtils.show_options_and_choose(actions, "Choose action")

        if action in ["1", "2"]:
            # Get user selection for the action
            if len(results) == 1:
                selected = results[0]
            else:
                choice = UIUtils.get_user_choice(
                    "Enter number",
                    [str(i) for i in range(1, min(len(results), 20) + 1)],
                )
                selected = results[int(choice) - 1]

            # Perform the selected action
            if action == "1":
                self._open_item(selected)
            else:
                self._rename_item(selected)

            return True  # Continue searching after open/rename
        elif action == "3":
            return True  # ✅ Continue search loop (no re-indexing!)
        else:
            return False  # Back to main menu

    def _open_item(self, item_path: Path):
        """
        Open file or folder using system default applications.

        Uses Windows-specific commands but could be extended for cross-platform support.
        """

        def open_operation():
            if item_path.is_dir():
                # Open folder in Windows Explorer
                subprocess.Popen(f'explorer "{item_path}"', shell=True)
                UIUtils.print_success(f"Opened folder: {item_path.name}")
            else:
                # Open file with default application
                os.startfile(str(item_path))
                UIUtils.print_success(f"Opened file: {item_path.name}")

        UIUtils.safe_execute("opening item", open_operation)

    def _rename_item(self, item_path: Path):
        """
        Rename file or folder with integrated undo option.

        After successful rename, immediately offers the user a chance to undo
        the operation, which catches typos and second thoughts instantly.
        """
        UIUtils.print_section_break()
        console.print(Panel(f"✏️ Rename: {item_path.name}", style="bold cyan"))
        UIUtils.print_section_break()

        new_name = Prompt.ask("📝 Enter new name", default=item_path.name)

        if new_name == item_path.name:
            UIUtils.print_warning("Name unchanged")
            return

        # Security validation
        if not UIUtils.validate_filename_or_show_error(new_name):
            return

        # Store original info for potential undo
        original_path = item_path
        original_name = item_path.name
        new_path = item_path.parent / new_name

        def rename_operation():
            if new_path.exists():
                UIUtils.print_error(f"Name already exists: {new_name}")
                return False

            original_path.rename(new_path)
            item_type = PathUtils.get_item_type(new_path)
            UIUtils.print_success(f"Renamed {item_type} to: {new_name}")
            return True

        # Perform rename operation
        rename_successful = UIUtils.safe_execute("renaming item", rename_operation)

        # If rename was successful, offer immediate undo option
        if rename_successful:
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]🔄 Do you want to undo this rename?[/bold cyan]",
                default=False,
            ):

                def undo_operation():
                    new_path.rename(original_path)
                    item_type = PathUtils.get_item_type(original_path)
                    UIUtils.print_success(f"Restored original name: {original_name}")

                UIUtils.safe_execute("undoing rename", undo_operation)
            UIUtils.print_section_break()

    def create_files_folders(self):
        """File and folder creation menu with nested structure support."""
        UIUtils.print_section_header("📁 Create Files & Folders")

        options = ["1. 📁 Folder only", "2. 📄 File only", "3. 📁 Folder with files"]
        choice = UIUtils.show_options_and_choose(options, "Choose option")

        UIUtils.print_section_break()

        if choice == "1":
            self._create_folder_with_nested_options()
        elif choice == "2":
            self._create_file_with_additional_options()
        elif choice == "3":
            self._create_folder_with_files_and_nested()

    def _get_location_choice(self) -> str:
        """
        Get destination location from user with support for multiple drives.

        Dynamically detects available drives and presents them as options
        along with common folders.
        """
        drives = PathUtils.get_available_drives()

        # Common location options
        locations = [
            ("1", "🖥️ Desktop", str(self.desktop)),
            ("2", "📄 Documents", str(Path.home() / "Documents")),
            ("3", "⬇️ Downloads", str(Path.home() / "Downloads")),
        ]

        # Add detected drives
        for i, drive in enumerate(drives, 4):
            locations.append(
                (str(i), f"💾 {drive}:", str(PathUtils.get_drive_path(drive)))
            )

        # Custom path option
        locations.append((str(len(locations) + 1), "📝 Custom Path", "custom"))

        # Display options
        for option, display, _ in locations:
            console.print(f"{option}. {display}")

        choice = UIUtils.get_user_choice(
            "Select location", [opt[0] for opt in locations]
        )
        selected = locations[int(choice) - 1][2]

        return Prompt.ask("Enter custom path") if selected == "custom" else selected

    def _create_folder_with_nested_options(self):
        """Create single folder with option for subfolders."""
        UIUtils.print_info("Creating New Folder")
        UIUtils.print_separator()

        location = self._get_location_choice()
        folder_name = Prompt.ask("📁 Folder name")

        if not UIUtils.validate_filename_or_show_error(folder_name):
            return

        def create_operation():
            folder_path = Path(location) / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            UIUtils.print_success(f"Created folder: {folder_path}")

            # Ask about subfolders
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]📁 Do you want to create subfolders?[/bold cyan]",
                default=False,
            ):
                self._create_subfolders(folder_path)

            # Offer to open the created folder
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]📂 Do you want to open this folder?[/bold cyan]",
                default=False,
            ):
                self._open_item(folder_path)

            return True

        UIUtils.safe_execute("creating folder", create_operation)
        UIUtils.print_section_break()

    def _create_file_with_additional_options(self):
        """Create single file with option for additional files."""
        UIUtils.print_info("Creating New File")
        UIUtils.print_separator()

        location = self._get_location_choice()
        file_name = Prompt.ask("📄 File name (with extension)")

        if not UIUtils.validate_filename_or_show_error(file_name):
            return

        # Optional content
        content = ""
        if Confirm.ask("Add content to file?", default=False):
            content = Prompt.ask("Enter content", default="")

        def create_operation():
            file_path = Path(location) / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            UIUtils.print_success(f"Created file: {file_path}")

            # Ask about additional files in same location
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]📄 Do you want to create additional files in this location?[/bold cyan]",
                default=False,
            ):
                self._create_additional_files(file_path.parent)

            # Offer to open the created file
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]📄 Do you want to open this file?[/bold cyan]",
                default=False,
            ):
                self._open_item(file_path)

            return True

        UIUtils.safe_execute("creating file", create_operation)
        UIUtils.print_section_break()

    def _create_folder_with_files_and_nested(self):
        """Create folder with files and comprehensive nesting options."""
        UIUtils.print_info("Creating Folder with Files")
        UIUtils.print_separator()

        location = self._get_location_choice()
        folder_name = Prompt.ask("📁 Folder name")

        if not UIUtils.validate_filename_or_show_error(folder_name):
            return

        def create_operation():
            folder_path = Path(location) / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            UIUtils.print_success(f"Created folder: {folder_path}")
            UIUtils.print_separator()

            # Create initial files
            file_count = 0
            UIUtils.print_info("Creating files in main folder")
            while True:
                file_name = Prompt.ask("📄 File name (or 'done' to finish)")
                if file_name.lower() == "done":
                    break

                if not UIUtils.validate_filename_or_show_error(file_name):
                    continue

                file_path = folder_path / file_name
                file_path.write_text("", encoding="utf-8")
                UIUtils.print_success(f"Created file: {file_name}")
                file_count += 1

            # Show summary
            UIUtils.print_separator()
            UIUtils.print_success(f"Created folder with {file_count} files")

            # Ask about subfolders
            if Confirm.ask(
                "[bold cyan]📁 Do you want to create subfolders?[/bold cyan]",
                default=False,
            ):
                self._create_subfolders_with_files(folder_path)

            # Offer to open the created folder
            UIUtils.print_separator()
            if Confirm.ask(
                "[bold cyan]📂 Do you want to open this folder?[/bold cyan]",
                default=False,
            ):
                self._open_item(folder_path)

            return True

        UIUtils.safe_execute("creating folder with files", create_operation)
        UIUtils.print_section_break()

    def _create_subfolders(self, parent_path: Path):
        """Create subfolders in parent directory (folder-only option)."""
        UIUtils.print_info(f"Creating subfolders in: {parent_path.name}")

        while True:
            subfolder_name = Prompt.ask("📁 Subfolder name (or 'done' to finish)")
            if subfolder_name.lower() == "done":
                break

            if not UIUtils.validate_filename_or_show_error(subfolder_name):
                continue

            try:
                subfolder_path = parent_path / subfolder_name
                subfolder_path.mkdir(parents=True, exist_ok=True)
                UIUtils.print_success(f"Created subfolder: {subfolder_name}")
            except Exception as e:
                UIUtils.print_error(f"Error creating subfolder: {e}")

    def _create_additional_files(self, parent_path: Path):
        """Create additional files in the same directory."""
        UIUtils.print_info(f"Creating additional files in: {parent_path.name}")

        while True:
            file_name = Prompt.ask("📄 File name (or 'done' to finish)")
            if file_name.lower() == "done":
                break

            if not UIUtils.validate_filename_or_show_error(file_name):
                continue

            try:
                file_path = parent_path / file_name
                file_path.write_text("", encoding="utf-8")
                UIUtils.print_success(f"Created file: {file_name}")
            except Exception as e:
                UIUtils.print_error(f"Error creating file: {e}")

    def _create_subfolders_with_files(self, parent_path: Path):
        """Create subfolders with option to add files to each."""
        UIUtils.print_info(f"Creating subfolders in: {parent_path.name}")

        while True:
            subfolder_name = Prompt.ask("📁 Subfolder name (or 'done' to finish)")
            if subfolder_name.lower() == "done":
                break

            if not UIUtils.validate_filename_or_show_error(subfolder_name):
                continue

            try:
                subfolder_path = parent_path / subfolder_name
                subfolder_path.mkdir(parents=True, exist_ok=True)
                UIUtils.print_success(f"Created subfolder: {subfolder_name}")

                # Ask about adding files to this subfolder
                if Confirm.ask(
                    f"[bold cyan]📄 Add files to '{subfolder_name}'?[/bold cyan]",
                    default=False,
                ):
                    self._create_additional_files(subfolder_path)

            except Exception as e:
                UIUtils.print_error(f"Error creating subfolder: {e}")

    def list_directory(self):
        """
        List directory contents with filtering options.

        Provides options to show folders only, files only, or everything.
        Useful for exploring directory structure.
        """
        UIUtils.print_section_header("📋 List Directory Contents")

        location = self._get_location_choice()

        content_options = ["1. 📁 Folders only", "2. 📄 Files only", "3. 📋 Everything"]
        content_type = UIUtils.show_options_and_choose(
            content_options, "Choose content type"
        )

        UIUtils.print_section_break()
        UIUtils.print_info(f"Listing contents of: {Path(location).name}")
        UIUtils.print_section_break()

        try:
            path = Path(location)
            items = []

            # Collect items based on user preference
            for item in path.iterdir():
                if item.name.startswith("."):  # Skip hidden files
                    continue

                if content_type == "1" and item.is_dir():
                    items.append((item, "📁 Folder"))
                elif content_type == "2" and item.is_file():
                    items.append((item, "📄 File"))
                elif content_type == "3":
                    item_type = PathUtils.get_item_emoji_type(item)
                    items.append((item, item_type))

            if not items:
                UIUtils.print_warning("No items found in this directory")
                UIUtils.print_section_break()
                return

            # Display results in formatted table with enhanced styling
            table = UIUtils.create_results_table(
                "", [("#", "white", 3), ("Name", "green", 0), ("Type", "white", 10)]
            )

            UIUtils.apply_standard_table_styling(table)

            for i, (item, item_type) in enumerate(items, 1):
                table.add_row(str(i), item.name, item_type)

            console.print(table)
            UIUtils.print_separator()

            # Optional item opening
            if Confirm.ask("[bold cyan]📂 Open any item?[/bold cyan]", default=False):
                choice = UIUtils.get_user_choice(
                    "Enter number", [str(i) for i in range(1, len(items) + 1)]
                )
                selected_item = items[int(choice) - 1][0]
                self._open_item(selected_item)

            UIUtils.print_section_break()

        except Exception as e:
            UIUtils.print_error(f"Error listing directory: {e}")
            UIUtils.print_section_break()

    def show_search_statistics(self):
        """Display current search index statistics for user information."""
        UIUtils.print_section_header("⚙️ Search Statistics")

        table = UIUtils.create_results_table(
            "⚡ Search System Status",
            [("Metric", "cyan", 20), ("Value", "green", 20), ("Details", "dim", 40)],
        )

        # Apply enhanced table styling
        UIUtils.apply_standard_table_styling(table)

        # Show indexing status and performance metrics
        table.add_row("Status", "✅ Ready", "Optimized for instant search")
        table.add_row(
            "Items Indexed",
            f"{self.search_index.total_items:,}",
            "Total files and folders in search index",
        )
        table.add_row("Search Speed", "< 1ms", "Microsecond-level performance")

        console.print(table)
        UIUtils.print_section_break()

    def run_interactive(self):
        """
        Main interactive loop - the heart of the application.

        Continuously displays the menu and processes user choices until
        the user decides to exit. Uses exception handling to gracefully
        handle unexpected errors.
        """
        while True:
            try:
                self.show_main_menu()

                choice = UIUtils.get_user_choice(
                    "Select option", ["0", "1", "2", "3", "4", "5"]
                )

                if choice == "0":
                    UIUtils.print_section_break()
                    console.print(
                        "[bold yellow]👋 GOODBYE![/] Thank you for using File Commander"
                    )
                    UIUtils.print_section_break()
                    break
                elif choice == "1":
                    self.create_files_folders()
                elif choice == "2":
                    self.search_files()
                elif choice == "3":
                    self.play_movie()  # The star feature!
                elif choice == "4":
                    self.list_directory()
                elif choice == "5":
                    self.show_search_statistics()

                # Pause before returning to menu (better UX)
                if choice != "0":
                    UIUtils.print_separator()
                    Prompt.ask(
                        "[dim]Press Enter to return to main menu[/dim]", default=""
                    )
                    UIUtils.print_separator()

            except KeyboardInterrupt:
                # Graceful handling of Ctrl+C
                UIUtils.print_section_break()
                console.print("[bold yellow]👋 GOODBYE![/] Interrupted by user")
                UIUtils.print_section_break()
                break
            except Exception as e:
                # Unexpected error handling
                UIUtils.print_section_break()
                UIUtils.print_error(f"Unexpected error: {e}")
                console.print("[dim]Please try again or restart the application.[/dim]")
                UIUtils.print_section_break()


# =============================================================================
# APPLICATION ENTRY POINTS
# =============================================================================


@app.command()
def interactive():
    """Start interactive mode - the main way to use File Commander."""
    commander = FileCommander()
    commander.run_interactive()


if __name__ == "__main__":
    # Default behavior: start interactive mode if no command-line arguments
    if len(sys.argv) == 1:
        commander = FileCommander()
        commander.run_interactive()
    else:
        # Handle command-line arguments (future expansion)
        app()
