"""
Streamlit Components for Intelligent Response Formatting
Handles rendering of tables, charts, and other formatted responses
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json

from services.intelligent_response_formatter import ResponseType, ChartType
from utils.logger import get_logger

logger = get_logger("streamlit_components")


class StreamlitComponentRenderer:
    """Renders intelligent response components in Streamlit"""
    
    def __init__(self):
        pass
    
    def render_formatted_response(self, response_metadata: Dict[str, Any]) -> None:
        """Render formatted response with appropriate Streamlit components"""
        try:
            intelligent_formatting = response_metadata.get("intelligent_formatting", {})
            
            if not intelligent_formatting:
                return
            
            response_format_type = intelligent_formatting.get("response_format_type", "plain_text")
            chart_data = intelligent_formatting.get("chart_data")
            table_data = intelligent_formatting.get("table_data")
            streamlit_components = intelligent_formatting.get("streamlit_components", [])
            
            # Render based on response type
            if response_format_type == ResponseType.TABLE.value and table_data:
                self._render_table(table_data)
            
            elif response_format_type == ResponseType.CHART.value and chart_data:
                self._render_chart(chart_data)
            
            elif response_format_type == ResponseType.COMPARISON.value:
                self._render_comparison_layout()
            
            # Render additional Streamlit components
            for component in streamlit_components:
                self._render_component(component)
                
        except Exception as e:
            logger.error(f"Failed to render formatted response: {e}")
    
    def _render_table(self, table_data: Dict[str, Any]) -> None:
        """Render table data using Streamlit's native table components"""
        try:
            if table_data:
                df = pd.DataFrame(table_data)
                
                # Use Streamlit's enhanced dataframe display
                st.subheader("📊 Data Table")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add download button for the data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="data_table.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            logger.error(f"Failed to render table: {e}")
            st.error("Failed to render table data")
    
    def _render_chart(self, chart_data: Dict[str, Any]) -> None:
        """Render chart using Streamlit's plotting capabilities"""
        try:
            chart_type = chart_data.get("chart_type", "bar_chart")
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            title = chart_data.get("title", "Data Visualization")
            
            if not labels or not values:
                return
            
            st.subheader("📈 Data Visualization")
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                "Category": labels,
                "Value": values
            })
            
            # Render different chart types
            if chart_type == ChartType.BAR_CHART.value:
                st.bar_chart(df.set_index("Category"))
                
            elif chart_type == ChartType.LINE_CHART.value:
                st.line_chart(df.set_index("Category"))
                
            elif chart_type == ChartType.AREA_CHART.value:
                st.area_chart(df.set_index("Category"))
                
            elif chart_type == ChartType.PIE_CHART.value:
                # Use Plotly for pie chart
                fig = px.pie(df, values="Value", names="Category", title=title)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == ChartType.SCATTER_PLOT.value:
                # Use Plotly for scatter plot
                fig = px.scatter(df, x="Category", y="Value", title=title)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == ChartType.HISTOGRAM.value:
                # Use Plotly for histogram
                fig = px.histogram(df, x="Value", title=title)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Default to bar chart
                st.bar_chart(df.set_index("Category"))
            
            # Add chart insights
            if len(values) > 1:
                max_val = max(values)
                min_val = min(values)
                avg_val = sum(values) / len(values)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Maximum", f"{max_val:.2f}")
                with col2:
                    st.metric("Minimum", f"{min_val:.2f}")
                with col3:
                    st.metric("Average", f"{avg_val:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to render chart: {e}")
            st.error("Failed to render chart data")
    
    def _render_comparison_layout(self) -> None:
        """Render comparison layout with side-by-side columns"""
        try:
            st.subheader("⚖️ Comparison View")
            st.info("This response contains comparison data. The information has been formatted for easy comparison.")
            
        except Exception as e:
            logger.error(f"Failed to render comparison layout: {e}")
    
    def _render_component(self, component: Dict[str, Any]) -> None:
        """Render individual Streamlit component"""
        try:
            component_type = component.get("type")
            
            if component_type == "dataframe":
                data = component.get("data", {})
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=component.get("use_container_width", False))
            
            elif component_type == "chart":
                chart_type = component.get("chart_type")
                data = component.get("data", {})
                self._render_chart(data)
            
            elif component_type == "metric":
                label = component.get("label", "Metric")
                value = component.get("value", 0)
                delta = component.get("delta")
                st.metric(label, value, delta)
            
            elif component_type == "progress":
                value = component.get("value", 0)
                st.progress(value)
            
            elif component_type == "alert":
                message = component.get("message", "")
                alert_type = component.get("alert_type", "info")
                
                if alert_type == "success":
                    st.success(message)
                elif alert_type == "warning":
                    st.warning(message)
                elif alert_type == "error":
                    st.error(message)
                else:
                    st.info(message)
                    
        except Exception as e:
            logger.error(f"Failed to render component: {e}")
    
    def render_response_type_indicator(self, response_format_type: str, confidence: float = 0.0) -> None:
        """Render indicator showing the detected response type"""
        try:
            # Map response types to emojis and descriptions
            type_info = {
                ResponseType.TABLE.value: ("📊", "Table Format", "Data organized in tabular format"),
                ResponseType.CHART.value: ("📈", "Chart Format", "Data visualized as charts/graphs"),
                ResponseType.LIST.value: ("📝", "List Format", "Information organized as lists"),
                ResponseType.CODE_BLOCK.value: ("💻", "Code Format", "Technical content with syntax highlighting"),
                ResponseType.STRUCTURED_TEXT.value: ("📄", "Structured Text", "Well-organized text with headings"),
                ResponseType.COMPARISON.value: ("⚖️", "Comparison Format", "Side-by-side comparison layout"),
                ResponseType.PROCESS_FLOW.value: ("🔄", "Process Flow", "Step-by-step process or workflow"),
                ResponseType.SUMMARY.value: ("📋", "Summary Format", "Concise summary with key points"),
                ResponseType.PLAIN_TEXT.value: ("📝", "Plain Text", "Standard text format")
            }
            
            emoji, title, description = type_info.get(response_format_type, ("📝", "Text Format", "Standard text format"))
            
            # Show format indicator with confidence
            with st.expander(f"{emoji} Response Format: {title}", expanded=False):
                st.write(f"**Format:** {description}")
                if confidence > 0:
                    st.write(f"**Confidence:** {confidence:.1%}")
                    
                    # Show confidence bar
                    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.progress(confidence)
                
        except Exception as e:
            logger.error(f"Failed to render response type indicator: {e}")
    
    def render_formatting_debug_info(self, formatting_metadata: Dict[str, Any]) -> None:
        """Render debug information about the formatting process"""
        try:
            if not formatting_metadata:
                return
                
            with st.expander("🔧 Formatting Debug Info", expanded=False):
                st.json(formatting_metadata)
                
        except Exception as e:
            logger.error(f"Failed to render formatting debug info: {e}")


# Global instance
streamlit_component_renderer = StreamlitComponentRenderer()