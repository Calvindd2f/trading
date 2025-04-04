import logging
from typing import List, Tuple, Dict, Optional, Set, Any
import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import time
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import networkx as nx
from scipy.sparse import csr_matrix
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import ta
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingNLPAnalyzer:
    """A class for performing advanced NLP analysis on trading-related text data."""
    
    def __init__(self, 
                 spacy_model: str = 'en_core_web_sm',
                 max_workers: int = 4,
                 cache_size: int = 128,
                 use_gpu: bool = False):
        """
        Initialize the trading NLP analyzer.

        Args:
            spacy_model (str): Name of the spaCy model to use
            max_workers (int): Maximum number of worker threads
            cache_size (int): Size of the LRU cache for text processing
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        # Load spaCy model with GPU if available
        if use_gpu and spacy.prefer_gpu():
            spacy.require_gpu()
        self.nlp = spacy.load(spacy_model)
        
        # Enhanced trading-specific patterns
        self.price_pattern = re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d+)?')
        self.percentage_pattern = re.compile(r'[-+]?\d+(?:\.\d+)?%')
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}')
        self.volume_pattern = re.compile(r'volume\s*:\s*\d+(?:,\d{3})*(?:\.\d+)?')
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?')
        
        # Enhanced trading-specific keywords
        self.trading_keywords = {
            'price_action': [
                'support', 'resistance', 'breakout', 'breakdown', 'trend',
                'consolidation', 'range', 'channel', 'wedge', 'flag'
            ],
            'indicators': [
                'rsi', 'macd', 'moving average', 'volume', 'stochastic',
                'bollinger bands', 'fibonacci', 'ichimoku', 'vwap', 'atr'
            ],
            'sentiment': [
                'bullish', 'bearish', 'neutral', 'overbought', 'oversold',
                'accumulation', 'distribution', 'fomo', 'fud', 'pump'
            ],
            'patterns': [
                'head and shoulders', 'double top', 'double bottom', 'triangle',
                'cup and handle', 'ascending triangle', 'descending triangle',
                'pennant', 'symmetrical triangle', 'falling wedge'
            ],
            'timeframes': [
                'daily', 'weekly', 'monthly', 'hourly', 'minute',
                '15m', '30m', '1h', '4h', '1d'
            ],
            'risk_management': [
                'stop loss', 'take profit', 'risk reward', 'position size',
                'trailing stop', 'break even', 'risk management'
            ]
        }
        
        # Technical indicators configuration
        self.indicators = {
            'rsi': {'window': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'window': 20, 'std': 2},
            'sma': {'short': 20, 'medium': 50, 'long': 200}
        }
        
        self.max_workers = max_workers
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize topic modeling
        self.lda = LatentDirichletAllocation(
            n_components=10,
            learning_method='online',
            random_state=42
        )
        
        # Initialize word embeddings
        self.word2vec = None
        self.phraser = None
        
        # Initialize caches
        self._clean_text = lru_cache(maxsize=cache_size)(self._clean_text)
        self._analyze_sentiment = lru_cache(maxsize=cache_size)(self._analyze_sentiment)
        self._extract_keywords = lru_cache(maxsize=cache_size)(self._extract_keywords)
        self._extract_entities = lru_cache(maxsize=cache_size)(self._extract_entities)
        self._extract_trading_signals = lru_cache(maxsize=cache_size)(self._extract_trading_signals)
        self._calculate_technical_indicators = lru_cache(maxsize=cache_size)(self._calculate_technical_indicators)

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
            
        # Enhanced text cleaning
        text = re.sub(r'http\S+|@\S+', '', text)  # URLs and mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Non-alphabetic
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = text.lower().strip()
        return text

    def _analyze_sentiment(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """
        Enhanced sentiment analysis with aspect-based sentiment.
        
        Returns:
            Tuple[float, float, Dict[str, float]]: (polarity, subjectivity, aspect_sentiments)
        """
        try:
            analysis = TextBlob(text)
            doc = self.nlp(text)
            
            # Aspect-based sentiment
            aspect_sentiments = {}
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    aspect = token.lemma_
                    aspect_text = f"{aspect} is good"
                    aspect_analysis = TextBlob(aspect_text)
                    aspect_sentiments[aspect] = aspect_analysis.sentiment.polarity
            
            return (analysis.sentiment.polarity, 
                   analysis.sentiment.subjectivity,
                   aspect_sentiments)
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return 0.0, 0.0, {}

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        try:
            doc = self.nlp(text)
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
            return dict(entities)
        except Exception as e:
            logging.error(f"Error in entity extraction: {e}")
            return {}

    def _extract_keywords(self, text: str, n: int = 10) -> List[Tuple[str, int]]:
        """Extract keywords with enhanced filtering."""
        try:
            doc = self.nlp(text)
            keywords = [
                token.lemma_ for token in doc 
                if (token.is_alpha and 
                    not token.is_stop and 
                    len(token) > 2 and
                    token.pos_ in ['NOUN', 'PROPN', 'ADJ'])
            ]
            return Counter(keywords).most_common(n)
        except Exception as e:
            logging.error(f"Error in keyword extraction: {e}")
            return []

    def _train_word_embeddings(self, texts: List[str]) -> None:
        """Train Word2Vec model on the corpus."""
        try:
            # Prepare text for Word2Vec
            sentences = [text.split() for text in texts]
            
            # Train phrase detector
            phrases = Phrases(sentences, min_count=5, threshold=10)
            self.phraser = Phraser(phrases)
            
            # Train Word2Vec
            self.word2vec = Word2Vec(
                self.phraser[sentences],
                vector_size=100,
                window=5,
                min_count=5,
                workers=self.max_workers
            )
        except Exception as e:
            logging.error(f"Error training word embeddings: {e}")

    def _extract_topics(self, texts: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Extract topics using LDA and NMF.
        
        Returns:
            Tuple[np.ndarray, List[List[str]]]: (topic_distributions, topic_keywords)
        """
        try:
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # LDA topic modeling
            lda_topics = self.lda.fit_transform(tfidf_matrix)
            
            # NMF topic modeling
            nmf = NMF(n_components=10, random_state=42)
            nmf_topics = nmf.fit_transform(tfidf_matrix)
            
            # Combine topics
            combined_topics = np.hstack([lda_topics, nmf_topics])
            
            # Extract topic keywords
            feature_names = self.vectorizer.get_feature_names_out()
            topic_keywords = []
            for topic_idx, topic in enumerate(self.lda.components_):
                top_keywords = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                topic_keywords.append(top_keywords)
            
            return combined_topics, topic_keywords
        except Exception as e:
            logging.error(f"Error in topic modeling: {e}")
            return np.array([]), []

    def _analyze_text_network(self, texts: List[str]) -> nx.Graph:
        """Analyze text relationships using network analysis."""
        try:
            G = nx.Graph()
            
            # Add nodes (keywords)
            all_keywords = set()
            for text in texts:
                keywords = [k for k, _ in self._extract_keywords(text)]
                all_keywords.update(keywords)
            
            G.add_nodes_from(all_keywords)
            
            # Add edges (co-occurrences)
            for text in texts:
                keywords = [k for k, _ in self._extract_keywords(text)]
                for i, k1 in enumerate(keywords):
                    for k2 in keywords[i+1:]:
                        if G.has_edge(k1, k2):
                            G[k1][k2]['weight'] += 1
                        else:
                            G.add_edge(k1, k2, weight=1)
            
            return G
        except Exception as e:
            logging.error(f"Error in network analysis: {e}")
            return nx.Graph()

    def _process_batch(self, texts: List[str]) -> pd.DataFrame:
        """Process a batch of texts with enhanced features."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Clean texts
            cleaned_texts = list(executor.map(self._clean_text, texts))
            
            # Analyze sentiment
            sentiment_results = list(executor.map(self._analyze_sentiment, cleaned_texts))
            polarities, subjectivities, aspect_sentiments = zip(*sentiment_results)
            
            # Extract entities and keywords
            entities = list(executor.map(self._extract_entities, cleaned_texts))
            keywords = list(executor.map(self._extract_keywords, cleaned_texts))
            
            # Create DataFrame
            return pd.DataFrame({
                'text': texts,
                'clean_text': cleaned_texts,
                'polarity': polarities,
                'subjectivity': subjectivities,
                'aspect_sentiments': aspect_sentiments,
                'entities': entities,
                'keywords': keywords
            })

    def process_text_data(self, 
                         df: pd.DataFrame, 
                         text_column: str = 'text',
                         batch_size: int = 1000) -> pd.DataFrame:
        """Process text data with enhanced analysis."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        start_time = time.time()
        processed_batches = []
        texts = df[text_column].tolist()
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            processed_batch = self._process_batch(batch)
            processed_batches.append(processed_batch)
            
            if (i + batch_size) % (batch_size * 5) == 0:
                logging.info(f"Processed {i + batch_size} texts...")

        # Combine batches
        result = pd.concat(processed_batches, ignore_index=True)
        
        # Train word embeddings
        self._train_word_embeddings(result['clean_text'].tolist())
        
        # Extract topics
        topics, topic_keywords = self._extract_topics(result['clean_text'].tolist())
        result['topic_distribution'] = list(topics)
        result['topic_keywords'] = [topic_keywords] * len(result)
        
        # Analyze text network
        text_network = self._analyze_text_network(result['clean_text'].tolist())
        result['network_centrality'] = [
            nx.degree_centrality(text_network).get(k, 0)
            for keywords in result['keywords']
            for k, _ in keywords
        ]
        
        logging.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
        return result

    def get_entity_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Get summary of named entities."""
        entity_counts = defaultdict(Counter)
        for entities in df['entities']:
            for entity_type, entity_list in entities.items():
                entity_counts[entity_type].update(entity_list)
        return {k: dict(v) for k, v in entity_counts.items()}

    def get_topic_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of topic modeling results."""
        topic_distributions = np.array(df['topic_distribution'].tolist())
        return {
            'dominant_topics': np.argmax(topic_distributions, axis=1),
            'topic_diversity': np.mean(np.std(topic_distributions, axis=1)),
            'topic_keywords': df['topic_keywords'].iloc[0]
        }

    def get_network_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of text network analysis."""
        G = self._analyze_text_network(df['clean_text'].tolist())
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'clustering_coefficient': nx.average_clustering(G),
            'density': nx.density(G)
        }

    def _extract_trading_signals(self, text: str) -> Dict[str, Any]:
        """Extract trading-specific signals from text."""
        try:
            signals = {
                'prices': [],
                'percentages': [],
                'tickers': [],
                'keywords': defaultdict(int),
                'timeframes': set(),
                'sentiment_score': 0.0
            }
            
            # Extract prices, percentages, and tickers
            signals['prices'] = self.price_pattern.findall(text)
            signals['percentages'] = self.percentage_pattern.findall(text)
            signals['tickers'] = self.ticker_pattern.findall(text)
            
            # Extract trading keywords
            text_lower = text.lower()
            for category, keywords in self.trading_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        signals['keywords'][category] += 1
            
            # Extract timeframes
            for timeframe in self.trading_keywords['timeframes']:
                if timeframe in text_lower:
                    signals['timeframes'].add(timeframe)
            
            # Calculate sentiment score
            sentiment = self._analyze_sentiment(text)
            signals['sentiment_score'] = sentiment[0]  # polarity
            
            return signals
        except Exception as e:
            logging.error(f"Error extracting trading signals: {e}")
            return {}

    def _analyze_trading_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze trading patterns and correlations."""
        try:
            patterns = {
                'price_movements': [],
                'sentiment_correlations': [],
                'keyword_cooccurrences': defaultdict(int),
                'timeframe_distribution': defaultdict(int)
            }
            
            for text in texts:
                signals = self._extract_trading_signals(text)
                
                # Analyze price movements
                if signals['prices'] and signals['percentages']:
                    price = float(signals['prices'][0].replace('$', '').replace(',', ''))
                    percentage = float(signals['percentages'][0].strip('%'))
                    patterns['price_movements'].append((price, percentage))
                
                # Analyze sentiment correlations
                if signals['sentiment_score'] != 0:
                    patterns['sentiment_correlations'].append(signals['sentiment_score'])
                
                # Analyze keyword co-occurrences
                for cat1, count1 in signals['keywords'].items():
                    for cat2, count2 in signals['keywords'].items():
                        if cat1 != cat2 and count1 > 0 and count2 > 0:
                            patterns['keyword_cooccurrences'][(cat1, cat2)] += 1
                
                # Analyze timeframe distribution
                for timeframe in signals['timeframes']:
                    patterns['timeframe_distribution'][timeframe] += 1
            
            return patterns
        except Exception as e:
            logging.error(f"Error analyzing trading patterns: {e}")
            return {}

    def _calculate_technical_indicators(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate technical indicators for price analysis."""
        try:
            prices = pd.Series(prices)
            
            # RSI
            rsi = RSIIndicator(
                close=prices,
                window=self.indicators['rsi']['window']
            ).rsi()
            
            # MACD
            macd = MACD(
                close=prices,
                window_slow=self.indicators['macd']['slow'],
                window_fast=self.indicators['macd']['fast'],
                window_sign=self.indicators['macd']['signal']
            )
            
            # Bollinger Bands
            bollinger = BollingerBands(
                close=prices,
                window=self.indicators['bollinger']['window'],
                window_dev=self.indicators['bollinger']['std']
            )
            
            # Moving Averages
            sma_short = SMAIndicator(
                close=prices,
                window=self.indicators['sma']['short']
            ).sma_indicator()
            
            sma_medium = SMAIndicator(
                close=prices,
                window=self.indicators['sma']['medium']
            ).sma_indicator()
            
            sma_long = SMAIndicator(
                close=prices,
                window=self.indicators['sma']['long']
            ).sma_indicator()
            
            return {
                'rsi': rsi.iloc[-1],
                'macd': macd.macd().iloc[-1],
                'macd_signal': macd.macd_signal().iloc[-1],
                'macd_hist': macd.macd_diff().iloc[-1],
                'bb_upper': bollinger.bollinger_hband().iloc[-1],
                'bb_middle': bollinger.bollinger_mavg().iloc[-1],
                'bb_lower': bollinger.bollinger_lband().iloc[-1],
                'sma_short': sma_short.iloc[-1],
                'sma_medium': sma_medium.iloc[-1],
                'sma_long': sma_long.iloc[-1]
            }
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return {}

    def _analyze_price_action(self, 
                            prices: List[float],
                            volume: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze price action and generate trading signals."""
        try:
            indicators = self._calculate_technical_indicators(prices)
            
            # Calculate price momentum
            price_changes = np.diff(prices)
            momentum = np.mean(price_changes[-5:])  # 5-period momentum
            
            # Calculate volatility
            volatility = np.std(price_changes[-20:])  # 20-period volatility
            
            # Calculate support and resistance levels
            recent_prices = prices[-20:]
            support = min(recent_prices)
            resistance = max(recent_prices)
            
            # Generate trading signals
            signals = {
                'trend': 'up' if momentum > 0 else 'down',
                'momentum_strength': abs(momentum) / volatility if volatility > 0 else 0,
                'volatility': volatility,
                'support': support,
                'resistance': resistance,
                'indicators': indicators
            }
            
            # Add volume analysis if available
            if volume is not None:
                volume_ma = np.mean(volume[-20:])
                signals['volume_trend'] = 'increasing' if volume[-1] > volume_ma else 'decreasing'
                signals['volume_ratio'] = volume[-1] / volume_ma if volume_ma > 0 else 1
            
            return signals
        except Exception as e:
            logging.error(f"Error analyzing price action: {e}")
            return {}

    def generate_trading_recommendations(self,
                                       prices: List[float],
                                       volume: Optional[List[float]] = None,
                                       sentiment: Optional[float] = None) -> Dict[str, Any]:
        """Generate trading recommendations based on multiple factors."""
        try:
            # Analyze price action
            price_signals = self._analyze_price_action(prices, volume)
            
            # Calculate recommendation score
            score = 0.0
            recommendations = []
            
            # Technical Analysis Factors
            if price_signals['indicators']:
                # RSI signals
                rsi = price_signals['indicators']['rsi']
                if rsi < self.indicators['rsi']['oversold']:
                    score += 0.3
                    recommendations.append('RSI indicates oversold conditions')
                elif rsi > self.indicators['rsi']['overbought']:
                    score -= 0.3
                    recommendations.append('RSI indicates overbought conditions')
                
                # MACD signals
                macd = price_signals['indicators']['macd']
                macd_signal = price_signals['indicators']['macd_signal']
                if macd > macd_signal:
                    score += 0.2
                    recommendations.append('MACD shows bullish crossover')
                else:
                    score -= 0.2
                    recommendations.append('MACD shows bearish crossover')
                
                # Moving Average signals
                sma_short = price_signals['indicators']['sma_short']
                sma_long = price_signals['indicators']['sma_long']
                if sma_short > sma_long:
                    score += 0.2
                    recommendations.append('Short-term MA above long-term MA')
                else:
                    score -= 0.2
                    recommendations.append('Short-term MA below long-term MA')
            
            # Price Action Factors
            if price_signals['trend'] == 'up':
                score += 0.2
                recommendations.append('Price is in an uptrend')
            else:
                score -= 0.2
                recommendations.append('Price is in a downtrend')
            
            # Volume Analysis
            if 'volume_trend' in price_signals:
                if price_signals['volume_trend'] == 'increasing':
                    score += 0.1
                    recommendations.append('Volume is increasing')
                else:
                    score -= 0.1
                    recommendations.append('Volume is decreasing')
            
            # Sentiment Analysis
            if sentiment is not None:
                score += sentiment * 0.2
                recommendations.append(f'Sentiment score: {sentiment:.2f}')
            
            # Generate stop loss and take profit levels
            current_price = prices[-1]
            atr = price_signals['volatility'] * 2  # 2x ATR for stop loss
            stop_loss = current_price - atr
            take_profit = current_price + (atr * 2)  # 2:1 risk-reward ratio
            
            return {
                'score': score,
                'recommendation': 'BUY' if score > 0 else 'SELL',
                'confidence': abs(score),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signals': recommendations,
                'price_action': price_signals
            }
        except Exception as e:
            logging.error(f"Error generating trading recommendations: {e}")
            return {}

    def visualize_trading_analysis(self, 
                                 df: pd.DataFrame,
                                 output_dir: str = 'visualizations') -> None:
        """Create enhanced visualizations for trading analysis results."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Enhanced Price Action Chart
            if 'price' in df.columns:
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ))
                
                # Add technical indicators
                if 'sma_20' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color='blue')
                    ))
                
                if 'sma_50' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color='orange')
                    ))
                
                # Add Bollinger Bands
                if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['bb_upper'],
                        name='BB Upper',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['bb_lower'],
                        name='BB Lower',
                        line=dict(color='gray', dash='dash')
                    ))
                
                fig.update_layout(
                    title='Price Action with Technical Indicators',
                    yaxis_title='Price',
                    xaxis_title='Time'
                )
                fig.write_html(f'{output_dir}/price_action.html')
            
            # 2. Enhanced Sentiment Analysis
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x=df.index, y='polarity', label='Sentiment')
            if 'price' in df.columns:
                ax2 = plt.twinx()
                sns.lineplot(data=df, x=df.index, y='price', ax=ax2, color='red', label='Price')
                ax2.set_ylabel('Price')
            plt.title('Sentiment vs Price')
            plt.savefig(f'{output_dir}/sentiment_vs_price.png')
            plt.close()
            
            # 3. Trading Signals Heatmap
            signals_df = pd.DataFrame([self._extract_trading_signals(text) for text in df['text']])
            plt.figure(figsize=(12, 8))
            sns.heatmap(signals_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Trading Signals Correlation')
            plt.savefig(f'{output_dir}/signals_correlation.png')
            plt.close()
            
            # 4. Interactive Trading Dashboard
            dashboard = go.Figure()
            
            # Add price chart
            dashboard.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Add volume chart
            dashboard.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                yaxis='y2'
            ))
            
            # Add RSI
            dashboard.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                yaxis='y3'
            ))
            
            dashboard.update_layout(
                title='Trading Dashboard',
                yaxis=dict(title='Price'),
                yaxis2=dict(title='Volume', overlaying='y', side='right'),
                yaxis3=dict(title='RSI', overlaying='y', side='left', position=0.1),
                xaxis=dict(rangeslider=dict(visible=True))
            )
            
            dashboard.write_html(f'{output_dir}/trading_dashboard.html')
            
            # 5. Trading Recommendations
            recommendations = []
            for i in range(len(df) - 20, len(df)):
                prices = df['close'].iloc[i-20:i+1].tolist()
                volume = df['volume'].iloc[i-20:i+1].tolist() if 'volume' in df.columns else None
                sentiment = df['polarity'].iloc[i] if 'polarity' in df.columns else None
                
                rec = self.generate_trading_recommendations(prices, volume, sentiment)
                recommendations.append(rec)
            
            rec_df = pd.DataFrame(recommendations)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=rec_df, x=df.index[-len(recommendations):], y='score')
            plt.title('Trading Recommendation Score')
            plt.savefig(f'{output_dir}/recommendation_score.png')
            plt.close()
            
            logging.info(f"Enhanced visualizations saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error creating enhanced visualizations: {e}")

def main():
    """Example usage of the enhanced TradingNLPAnalyzer."""
    try:
        # Initialize analyzer
        analyzer = TradingNLPAnalyzer(
            spacy_model='en_core_web_sm',
            max_workers=4,
            use_gpu=True
        )
        
        # Load data
        df = pd.read_csv('data/text_data.csv')
        
        # Process data
        processed_df = analyzer.process_text_data(df)
        
        # Generate enhanced visualizations
        analyzer.visualize_trading_analysis(processed_df)
        
        # Generate trading recommendations
        prices = processed_df['close'].tolist()[-100:]  # Last 100 prices
        volume = processed_df['volume'].tolist()[-100:] if 'volume' in processed_df.columns else None
        sentiment = processed_df['polarity'].iloc[-1]
        
        recommendations = analyzer.generate_trading_recommendations(
            prices, volume, sentiment
        )
        
        logging.info("\nTrading Recommendations:")
        logging.info(f"Recommendation: {recommendations['recommendation']}")
        logging.info(f"Confidence: {recommendations['confidence']:.2f}")
        logging.info(f"Stop Loss: {recommendations['stop_loss']:.2f}")
        logging.info(f"Take Profit: {recommendations['take_profit']:.2f}")
        logging.info("\nSignals:")
        for signal in recommendations['signals']:
            logging.info(f"- {signal}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
