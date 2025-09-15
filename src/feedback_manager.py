
"""
Feedback Manager for Precision Farming

Comprehensive feedback management system for collecting, analyzing,
and acting on user feedback to improve the precision farming platform.

Features:
- Feedback collection and storage
- Sentiment analysis
- User experience tracking
- Feature usage analytics
- Improvement recommendations
- Feedback reporting

Author: Precision Farming Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
import uuid
import sqlite3
from collections import Counter, defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    USER_EXPERIENCE = "user_experience"
    GENERAL_FEEDBACK = "general_feedback"
    SUCCESS_STORY = "success_story"
    COMPLAINT = "complaint"
    SUGGESTION = "suggestion"


class Priority(Enum):
    """Feedback priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(Enum):
    """Feedback status"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REJECTED = "rejected"


class SentimentScore(Enum):
    """Sentiment analysis scores"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class UserProfile:
    """User profile information"""
    user_id: str
    name: str = "Anonymous"
    email: str = ""
    farm_size: float = 0.0
    location: str = ""
    primary_crops: List[str] = field(default_factory=list)
    experience_level: str = "beginner"  # beginner, intermediate, expert
    registration_date: str = ""
    last_active: str = ""
    total_feedback_count: int = 0


@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    user_id: str
    timestamp: str
    feedback_type: FeedbackType
    title: str
    description: str
    
    # Classification
    priority: Priority = Priority.MEDIUM
    status: Status = Status.NEW
    category: str = "general"
    
    # Ratings and scores
    overall_rating: Optional[int] = None  # 1-10 scale
    feature_ratings: Dict[str, int] = field(default_factory=dict)
    satisfaction_score: Optional[int] = None  # 1-5 scale
    sentiment_score: Optional[SentimentScore] = None
    
    # Context information
    page_url: str = ""
    user_agent: str = ""
    session_id: str = ""
    feature_used: str = ""
    
    # Processing information
    assigned_to: str = ""
    resolution_notes: str = ""
    resolution_date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    attachments: List[str] = field(default_factory=list)
    related_tickets: List[str] = field(default_factory=list)
    upvotes: int = 0
    downvotes: int = 0


@dataclass
class FeatureUsage:
    """Feature usage analytics"""
    feature_name: str
    usage_count: int
    unique_users: int
    average_session_duration: float
    success_rate: float
    error_count: int
    user_ratings: List[int]
    most_common_issues: List[str]


@dataclass
class UserExperience:
    """User experience metrics"""
    user_id: str
    session_count: int
    total_time_spent: float  # minutes
    features_used: List[str]
    success_rate: float
    error_count: int
    support_tickets: int
    satisfaction_trend: List[int]
    last_feedback_date: str


@dataclass
class FeedbackAnalytics:
    """Comprehensive feedback analytics"""
    total_feedback_count: int
    feedback_by_type: Dict[str, int]
    feedback_by_priority: Dict[str, int]
    feedback_by_status: Dict[str, int]
    average_rating: float
    sentiment_distribution: Dict[str, int]
    
    # Trends
    feedback_trend: List[Tuple[str, int]]  # (date, count)
    satisfaction_trend: List[Tuple[str, float]]  # (date, avg_satisfaction)
    
    # Feature analytics
    feature_feedback: Dict[str, FeatureUsage]
    most_requested_features: List[Tuple[str, int]]
    most_reported_bugs: List[Tuple[str, int]]
    
    # User analytics
    active_users: int
    user_retention_rate: float
    user_satisfaction_distribution: Dict[str, int]


class FeedbackManager:
    """
    Main class for managing user feedback and analytics
    """
    
    def __init__(self, db_path: str = "feedback.db", config: Optional[Dict] = None):
        """
        Initialize feedback manager
        
        Args:
            db_path: Path to SQLite database file
            config: Optional configuration dictionary
        """
        self.db_path = db_path
        self.config = config or self._load_default_config()
        
        # Initialize database
        self._init_database()
        
        # In-memory caches
        self.feedback_cache = {}
        self.user_cache = {}
        self.analytics_cache = {}
        
        logger.info("Feedback manager initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'auto_sentiment_analysis': True,
            'auto_categorization': True,
            'notification_email': 'admin@precisionfarming.com',
            'analytics_refresh_interval': 3600,  # seconds
            'max_feedback_age_days': 365,
            'sentiment_keywords': {
                'positive': ['great', 'excellent', 'love', 'amazing', 'helpful', 'easy'],
                'negative': ['terrible', 'awful', 'hate', 'difficult', 'confusing', 'slow'],
                'neutral': ['okay', 'average', 'fine', 'decent']
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    farm_size REAL,
                    location TEXT,
                    primary_crops TEXT,
                    experience_level TEXT,
                    registration_date TEXT,
                    last_active TEXT,
                    total_feedback_count INTEGER DEFAULT 0
                )
            ''')
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TEXT,
                    feedback_type TEXT,
                    title TEXT,
                    description TEXT,
                    priority TEXT,
                    status TEXT,
                    category TEXT,
                    overall_rating INTEGER,
                    satisfaction_score INTEGER,
                    sentiment_score TEXT,
                    page_url TEXT,
                    feature_used TEXT,
                    assigned_to TEXT,
                    resolution_notes TEXT,
                    resolution_date TEXT,
                    tags TEXT,
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Feature ratings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT,
                    feature_name TEXT,
                    rating INTEGER,
                    FOREIGN KEY (feedback_id) REFERENCES feedback (feedback_id)
                )
            ''')
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration REAL,
                    features_used TEXT,
                    errors_encountered INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
        
        logger.info("Database initialized successfully")
    
    def submit_feedback(self, feedback: FeedbackEntry) -> str:
        """
        Submit new feedback entry
        
        Args:
            feedback: FeedbackEntry object
            
        Returns:
            Feedback ID
        """
        # Generate ID if not provided
        if not feedback.feedback_id:
            feedback.feedback_id = str(uuid.uuid4())
        
        # Auto-analyze sentiment if enabled
        if self.config.get('auto_sentiment_analysis', True):
            feedback.sentiment_score = self._analyze_sentiment(feedback.description)
        
        # Auto-categorize if enabled
        if self.config.get('auto_categorization', True):
            feedback.category = self._categorize_feedback(feedback.description, feedback.feedback_type)
        
        # Set priority based on content
        feedback.priority = self._determine_priority(feedback)
        
        # Store in database
        self._store_feedback(feedback)
        
        # Update user statistics
        self._update_user_stats(feedback.user_id)
        
        # Clear analytics cache
        self.analytics_cache.clear()
        
        logger.info(f"Feedback submitted: {feedback.feedback_id}")
        
        return feedback.feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """
        Retrieve feedback by ID
        
        Args:
            feedback_id: Feedback identifier
            
        Returns:
            FeedbackEntry object or None
        """
        # Check cache first
        if feedback_id in self.feedback_cache:
            return self.feedback_cache[feedback_id]
        
        # Query database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM feedback WHERE feedback_id = ?
            ''', (feedback_id,))
            
            row = cursor.fetchone()
            if row:
                feedback = self._row_to_feedback(row)
                self.feedback_cache[feedback_id] = feedback
                return feedback
        
        return None
    
    def update_feedback_status(self, feedback_id: str, status: Status, 
                             resolution_notes: str = "") -> bool:
        """
        Update feedback status
        
        Args:
            feedback_id: Feedback identifier
            status: New status
            resolution_notes: Optional resolution notes
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                update_data = [status.value, feedback_id]
                query = "UPDATE feedback SET status = ?"
                
                if resolution_notes:
                    query += ", resolution_notes = ?"
                    update_data.insert(-1, resolution_notes)
                
                if status in [Status.RESOLVED, Status.CLOSED]:
                    query += ", resolution_date = ?"
                    update_data.insert(-1, datetime.now().isoformat())
                
                query += " WHERE feedback_id = ?"
                
                cursor.execute(query, update_data)
                conn.commit()
                
                # Update cache
                if feedback_id in self.feedback_cache:
                    self.feedback_cache[feedback_id].status = status
                    if resolution_notes:
                        self.feedback_cache[feedback_id].resolution_notes = resolution_notes
                
                logger.info(f"Feedback {feedback_id} status updated to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update feedback status: {e}")
            return False
    
    def get_feedback_by_user(self, user_id: str, limit: int = 50) -> List[FeedbackEntry]:
        """
        Get all feedback from a specific user
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of FeedbackEntry objects
        """
        feedback_list = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
