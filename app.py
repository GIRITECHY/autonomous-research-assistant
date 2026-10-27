import streamlit as st

from main import stream_research


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autonomous Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 17px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .stat-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        min-height: 110px;
    }

    .stat-number {
        font-size: 30px;
        font-weight: 800;
    }

    .stat-label {
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Autonomous Research Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'An agentic AI research system that searches, verifies, '
    'retrieves, analyzes and writes structured research reports.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success("🟢 System Ready")

    st.divider()

    st.subheader("🤖 Agent Pipeline")

    st.markdown(
        """
        🔍 **Researcher Agent**

        ↓

        🛡️ **Source Verification**

        ↓

        🧠 **RAG Retrieval**

        ↓

        📊 **Analyst Agent**

        ↓

        ✍️ **Writer Agent**
        """
    )

    st.divider()

    st.subheader("🧰 Technologies")

    st.markdown(
        """
        • LangGraph  
        • Google Gemini  
        • Tavily Search  
        • ChromaDB  
        • RAG  
        • Streamlit
        """
    )

    st.divider()

    st.subheader("🔬 Research Modes")

    st.markdown(
        """
        ⚡ **Quick**  
        Focused and faster research

        📚 **Standard**  
        Balanced research coverage

        🔎 **Deep**  
        Broader research coverage
        """
    )

    st.divider()

    st.caption(
        "Autonomous Research Assistant"
    )

    st.caption(
        "AI Agentic Development Internship"
    )


# ============================================================
# RESEARCH WORKSPACE
# ============================================================

st.subheader("📚 Research Workspace")


# ============================================================
# TOPIC INPUT
# ============================================================

topic = st.text_area(
    "Research Topic",
    placeholder=(
        "Example: Artificial Intelligence in Healthcare"
    ),
    height=110,
)


# ============================================================
# RESEARCH DEPTH
# ============================================================

st.markdown(
    "### 🔬 Research Depth"
)

depth_options = {
    "⚡ Quick": "quick",
    "📚 Standard": "standard",
    "🔎 Deep": "deep",
}


selected_depth_label = st.selectbox(
    "Choose how extensively the system should research the topic",
    options=list(
        depth_options.keys()
    ),
    index=1,
)


selected_depth = depth_options[
    selected_depth_label
]


# ============================================================
# DEPTH DESCRIPTION
# ============================================================

if selected_depth == "quick":

    st.info(
        "⚡ Quick Research — 3 search queries "
        "with focused source coverage."
    )

elif selected_depth == "standard":

    st.info(
        "📚 Standard Research — 5 search queries "
        "with balanced source coverage."
    )

else:

    st.info(
        "🔎 Deep Research — 8 search queries "
        "with broader source coverage."
    )


# ============================================================
# START BUTTON
# ============================================================

start_button = st.button(
    "🚀 Start Autonomous Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# START RESEARCH
# ============================================================

if start_button:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a research topic first."
        )

        st.stop()


    # ========================================================
    # RESEARCH CONFIGURATION
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Research Configuration"
    )

    config_col1, config_col2 = st.columns(
        2
    )


    with config_col1:

        st.markdown(
            f"**Research Topic:**  \n"
            f"{topic.strip()}"
        )


    with config_col2:

        st.markdown(
            f"**Research Depth:**  \n"
            f"{selected_depth_label}"
        )


    # ========================================================
    # AGENT WORKFLOW
    # ========================================================

    st.divider()

    st.subheader(
        "⚡ Autonomous Agent Workflow"
    )

    st.caption(
        "LangGraph is executing the research pipeline..."
    )


    stage_researcher = st.empty()

    stage_verification = st.empty()

    stage_rag = st.empty()

    stage_analyst = st.empty()

    stage_writer = st.empty()


    stage_researcher.info(
        "🔍 Researcher Agent — Waiting..."
    )

    stage_verification.info(
        "🛡️ Source Verification — Waiting..."
    )

    stage_rag.info(
        "🧠 RAG Retrieval — Waiting..."
    )

    stage_analyst.info(
        "📊 Analyst Agent — Waiting..."
    )

    stage_writer.info(
        "✍️ Writer Agent — Waiting..."
    )


    # ========================================================
    # FINAL STATE
    # ========================================================

    final_state = {
        "topic": topic.strip(),
        "depth": selected_depth,

        "search_queries": [],
        "query_count": 0,
        "total_search_results": 0,

        "research_results": [],
        "verified_sources": [],
        "retrieved_documents": [],

        "analysis": "",
        "final_report": "",
    }


    # ========================================================
    # STREAM LANGGRAPH
    # ========================================================

    try:

        for node_name, node_output in stream_research(
            topic,
            selected_depth,
        ):

            final_state.update(
                node_output
            )


            # =================================================
            # RESEARCHER
            # =================================================

            if node_name == "researcher":

                stage_researcher.success(
                    "🔍 Researcher Agent — Completed"
                )

                stage_verification.info(
                    "🛡️ Source Verification — Running..."
                )


            # =================================================
            # SOURCE VERIFICATION
            # =================================================

            elif node_name == "source_verification":

                stage_verification.success(
                    "🛡️ Source Verification — Completed"
                )

                stage_rag.info(
                    "🧠 RAG Retrieval — Running..."
                )


            # =================================================
            # RAG
            # =================================================

            elif node_name == "rag":

                stage_rag.success(
                    "🧠 RAG Retrieval — Completed"
                )

                stage_analyst.info(
                    "📊 Analyst Agent — Running..."
                )


            # =================================================
            # ANALYST
            # =================================================

            elif node_name == "analyst":

                stage_analyst.success(
                    "📊 Analyst Agent — Completed"
                )

                stage_writer.info(
                    "✍️ Writer Agent — Running..."
                )


            # =================================================
            # WRITER
            # =================================================

            elif node_name == "writer":

                stage_writer.success(
                    "✍️ Writer Agent — Completed"
                )


        # ====================================================
        # WORKFLOW COMPLETE
        # ====================================================

        st.success(
            "✅ Autonomous research workflow completed successfully!"
        )


        # ====================================================
        # GET RESULTS
        # ====================================================

        search_queries = final_state.get(
            "search_queries",
            []
        )

        query_count = final_state.get(
            "query_count",
            len(search_queries)
        )

        total_search_results = final_state.get(
            "total_search_results",
            0
        )

        research_results = final_state.get(
            "research_results",
            []
        )

        verified_sources = final_state.get(
            "verified_sources",
            []
        )

        retrieved_documents = final_state.get(
            "retrieved_documents",
            []
        )

        analysis = final_state.get(
            "analysis",
            ""
        )

        final_report = final_state.get(
            "final_report",
            ""
        )


        # ====================================================
        # RESEARCH ACTIVITY STATISTICS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Research Activity"
        )

        st.caption(
            "Live statistics collected from the completed "
            "LangGraph research run."
        )


        stat1, stat2, stat3 = st.columns(
            3
        )


        with stat1:

            st.metric(
                "🔍 Search Queries",
                query_count
            )


        with stat2:

            st.metric(
                "🌐 Search Results",
                total_search_results
            )


        with stat3:

            st.metric(
                "🛡️ Verified Sources",
                len(verified_sources)
            )


        stat4, stat5, stat6 = st.columns(
            3
        )


        with stat4:

            st.metric(
                "🧠 RAG Sources",
                len(retrieved_documents)
            )


        with stat5:

            st.metric(
                "🤖 Agents Completed",
                5
            )


        with stat6:

            st.metric(
                "🔬 Research Mode",
                selected_depth.upper()
            )


        # ====================================================
        # SEARCH QUERIES
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 Search Queries Used"
        )

        st.caption(
            "Queries generated by the Researcher Agent "
            "for this research task."
        )


        if search_queries:

            for index, query in enumerate(
                search_queries,
                1
            ):

                st.markdown(
                    f"**{index}.** {query}"
                )

        else:

            st.info(
                "No search queries were recorded."
            )


        # ====================================================
        # SOURCE PIPELINE SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "🔗 Source Pipeline"
        )

        pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(
            3
        )


        with pipeline_col1:

            st.metric(
                "Sources Selected",
                len(research_results)
            )


        with pipeline_col2:

            st.metric(
                "Technically Verified",
                len(verified_sources)
            )


        with pipeline_col3:

            st.metric(
                "Retrieved by RAG",
                len(retrieved_documents)
            )


        # ====================================================
        # VERIFIED SOURCES
        # ====================================================

        st.divider()

        st.subheader(
            "🔗 Verified Research Sources"
        )

        st.caption(
            "Sources that passed the system's technical validation."
        )


        if verified_sources:

            for index, source in enumerate(
                verified_sources,
                1
            ):

                title = source.get(
                    "title",
                    "Unknown Source"
                )

                url = source.get(
                    "url",
                    ""
                )

                content = source.get(
                    "content",
                    ""
                )


                with st.expander(
                    f"📄 {index}. {title}"
                ):

                    st.markdown(
                        f"**🔗 URL:** {url}"
                    )

                    st.success(
                        "Technical verification passed"
                    )


                    if content:

                        st.markdown(
                            "**Source Content Preview**"
                        )

                        st.write(
                            content[:1500]
                        )

        else:

            st.info(
                "No verified sources were returned."
            )


        # ====================================================
        # RAG RETRIEVAL
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 RAG Retrieved Research"
        )

        st.caption(
            "Relevant information retrieved from the "
            "current research run."
        )


        if retrieved_documents:

            for index, document in enumerate(
                retrieved_documents,
                1
            ):

                title = document.get(
                    "title",
                    "Unknown Source"
                )

                url = document.get(
                    "url",
                    ""
                )

                content = document.get(
                    "content",
                    ""
                )


                with st.expander(
                    f"🧠 Retrieved Source {index}: {title}"
                ):

                    st.markdown(
                        f"**🔗 URL:** {url}"
                    )

                    st.write(
                        content
                    )

        else:

            st.info(
                "No RAG documents were retrieved."
            )


        # ====================================================
        # AI ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Analyst Agent Output"
        )


        if analysis:

            with st.container(
                border=True
            ):

                st.markdown(
                    analysis
                )

        else:

            st.info(
                "No analysis was generated."
            )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        st.divider()

        st.subheader(
            "📄 Final Research Report"
        )


        if final_report:

            with st.container(
                border=True
            ):

                st.markdown(
                    final_report
                )


            # =================================================
            # DOWNLOAD OPTIONS
            # =================================================

            st.markdown(
                "### ⬇️ Download Report"
            )


            download_col1, download_col2 = st.columns(
                2
            )


            with download_col1:

                st.download_button(
                    label="📄 Download Markdown",
                    data=final_report,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


            with download_col2:

                st.download_button(
                    label="📝 Download Text",
                    data=final_report,
                    file_name="research_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )


        else:

            st.error(
                "❌ Final report was not generated."
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ The research workflow encountered an error."
        )

        st.exception(e)