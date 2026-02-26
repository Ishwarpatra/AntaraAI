import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
# import pymongo.errors # Import pymongo.errors - now handled by conftest

# No module-level patches here anymore. All handled in conftest.py

from core.agent import analyze_sentiment, should_request_selfie, crisis_node, agent, route_tools
from core.state import State
from langchain_core.messages import HumanMessage, AIMessage
from config.system_config import SystemConfig

# Mock GamificationManager for log_mood_tool
@pytest.fixture
def mock_gamification_manager():
    with patch('core.gamification.GamificationManager', autospec=True) as MockGamificationManager:
        instance = MockGamificationManager.return_value
        instance.log_mood_event.return_value = {
            "xp_gained": 10,
            "total_xp": 10,
            "current_streak": 1,
            "streak_message": "Streak started!"
        }
        instance.get_user_progress.return_value = {
            "xp": 100,
            "streak_days": 5,
            "last_mood_log_date": "2026-02-25"
        }
        yield instance

# Mock SystemConfig if needed, though for analyze_sentiment it's usually static
@pytest.fixture
def mock_system_config():
    original_config = SystemConfig.SENTIMENT_ANALYSIS_CONFIG
    with patch.dict(SystemConfig.SENTIMENT_ANALYSIS_CONFIG, original_config, clear=True):
        SystemConfig.SENTIMENT_ANALYSIS_CONFIG = {
            "CRITICAL": {"patterns": [r"kill myself"], "keywords": ["suicide"], "score_impact": 5},
            "WARNING": {"patterns": [r"feel down"], "keywords": ["depressed"], "score_impact": 3},
            "CAUTION": {"patterns": [], "keywords": ["sad"], "score_impact": 1}
        }
        yield SystemConfig
    SystemConfig.SENTIMENT_ANALYSIS_CONFIG = original_config # Restore original

class TestAnalyzeSentiment:
    def test_critical_sentiment(self, mock_system_config):
        assert analyze_sentiment("I want to kill myself") == "CRITICAL"
        assert analyze_sentiment("I'm planning suicide") == "CRITICAL"

    def test_warning_sentiment(self, mock_system_config):
        assert analyze_sentiment("I feel really down lately") == "WARNING"
        assert analyze_sentiment("I am so depressed") == "WARNING"

    def test_caution_sentiment(self, mock_system_config):
        assert analyze_sentiment("I am feeling a bit sad") == "CAUTION"

    def test_normal_sentiment(self, mock_system_config):
        assert analyze_sentiment("I am happy today") == "NORMAL"
        assert analyze_sentiment("The weather is nice") == "NORMAL"

    def test_mixed_sentiment(self, mock_system_config):
        # Should return the highest severity detected
        assert analyze_sentiment("I feel down but also a bit sad") == "WARNING"
        assert analyze_sentiment("I want to kill myself but I also feel happy") == "CRITICAL"

class TestShouldRequestSelfie:
    @patch('core.memory_manager.db')
    @patch('datetime.datetime')
    def test_no_selfie_if_recent(self, mock_datetime, mock_db):
        mock_db.selfie_requests.find.return_value = [{"timestamp": datetime.now() - timedelta(hours=1)}]
        mock_datetime.now.return_value = datetime.now()
        
        SystemConfig.SELFIE_REQUEST_INTERVAL_HOURS = 2
        state = State(messages=[HumanMessage(content="Hello")])
        config = {"configurable": {"user_id": "test_user"}}
        assert not should_request_selfie(state, config)

    @patch('core.memory_manager.db')
    @patch('datetime.datetime')
    def test_selfie_if_mood_keywords(self, mock_datetime, mock_db):
        mock_db.selfie_requests.find.return_value = [] # No recent selfie
        mock_datetime.now.return_value = datetime.now()
        
        SystemConfig.SELFIE_REQUEST_INTERVAL_HOURS = 0 # Ensure it's not blocked by interval
        state = State(messages=[HumanMessage(content="I'm feeling so depressed")])
        config = {"configurable": {"user_id": "test_user"}}
        assert should_request_selfie(state, config)

    @patch('core.memory_manager.db')
    @patch('datetime.datetime')
    @patch('random.random', return_value=0.05) # Force random chance success
    def test_selfie_if_random_chance(self, mock_random, mock_datetime, mock_db):
        mock_db.selfie_requests.find.return_value = [] # No recent selfie
        mock_datetime.now.return_value = datetime.now()
        
        SystemConfig.SELFIE_REQUEST_INTERVAL_HOURS = 0
        SystemConfig.SELFIE_RANDOM_CHANCE = 0.1
        state = State(messages=[HumanMessage(content="Hello")]) # No mood keywords
        config = {"configurable": {"user_id": "test_user"}}
        assert should_request_selfie(state, config)

class TestCrisisNode:
    @patch('core.tools.crisis_escalation_tool.invoke')
    @patch('core.memory_manager.db')
    @patch('core.agent.analyze_sentiment', return_value="CRITICAL")
    def test_critical_crisis_node(self, mock_analyze_sentiment, mock_db, mock_escalation_invoke):
        state = State(messages=[HumanMessage(content="I want to end it all.")])
        config = {"configurable": {"user_id": "test_user_crisis"}}
        
        result = crisis_node(state, config)
        
        mock_escalation_invoke.assert_called_once()
        mock_db.crisis_events.insert_one.assert_called_once()
        assert "immediate distress" in result["messages"][0].content
        assert any(tc.get("name") == "request_selfie_tool" for tc in result["messages"][0].tool_calls)
        assert result["next_node"] is None

    @patch('core.tools.crisis_escalation_tool.invoke')
    @patch('core.memory_manager.db')
    @patch('core.agent.analyze_sentiment', return_value="WARNING")
    def test_warning_crisis_node(self, mock_analyze_sentiment, mock_db, mock_escalation_invoke):
        state = State(messages=[HumanMessage(content="I'm really struggling.")])
        config = {"configurable": {"user_id": "test_user_crisis"}}
        
        result = crisis_node(state, config)
        
        mock_escalation_invoke.assert_called_once()
        mock_db.crisis_events.insert_one.assert_called_once()
        assert "concerned about what you've shared" in result["messages"][0].content
        assert not result["messages"][0].tool_calls # No selfie request for WARNING
        assert result["next_node"] is None

class TestAgentFunction:
    @patch('core.agent.analyze_sentiment', return_value="CRITICAL")
    def test_agent_triggers_crisis_node(self, mock_analyze_sentiment):
        state = State(messages=[HumanMessage(content="I'm in crisis!")])
        config = {"configurable": {"user_id": "test_user_agent"}}
        
        result = agent(state, config, MagicMock()) # model_with_tools is mocked
        
        assert result["next_node"] == "crisis_node"

    @patch('core.agent.should_request_selfie', return_value=True)
    def test_agent_requests_selfie(self, mock_should_request_selfie):
        state = State(messages=[HumanMessage(content="Hello")])
        config = {"configurable": {"user_id": "test_user_agent"}}
        
        result = agent(state, config, MagicMock())
        
        assert any(tc.get("name") == "request_selfie_tool" for tc in result["messages"][0].tool_calls)

    @patch('core.agent.analyze_sentiment', return_value="NORMAL")
    @patch('core.agent.should_request_selfie', return_value=False)
    def test_agent_invokes_model_with_tools(self, mock_should_request_selfie, mock_analyze_sentiment):
        mock_model_with_tools = MagicMock()
        mock_model_with_tools.invoke.return_value = AIMessage(content="Hello from AI!")
        
        state = State(messages=[HumanMessage(content="Normal message")])
        config = {"configurable": {"user_id": "test_user_agent"}}
        
        result = agent(state, config, mock_model_with_tools)
        
        mock_model_with_tools.invoke.assert_called_once()
        assert result["messages"][0].content == "Hello from AI!"

class TestRouteTools:
    def test_route_to_crisis_node(self):
        state = State(next_node="crisis_node", messages=[HumanMessage(content="Crisis message")])
        assert route_tools(state) == "crisis_node"

    def test_route_to_tools(self):
        state = State(messages=[AIMessage(content="Tool call", tool_calls=[{"name": "some_tool"}])])
        assert route_tools(state) == "tools"

    def test_route_to_end(self):
        state = State(messages=[AIMessage(content="Final message")])
        assert route_tools(state) == END

# --- New Test Class for Graph Transitions ---
class TestGraphTransitions:
    @pytest.fixture(autouse=True)
    def setup_graph_mocks(self):
        self.mock_model_with_tools = MagicMock()
        self.mock_model_with_tools.invoke.return_value = AIMessage(content="AI response")
        
        self.mock_load_memories = patch('core.agent.load_memories', return_value={"recall_memories": ["memory 1"]}).start()
        self.mock_agent = patch('core.agent.agent', side_effect=lambda state, config, model: {"messages": [model.invoke({"messages":state["messages"], "recall_memories": ""})], "next_node": None}).start()
        self.mock_route_tools = patch('core.agent.route_tools', return_value=END).start()
        self.mock_crisis_node = patch('core.agent.crisis_node', return_value={"messages": [AIMessage(content="Crisis handled")], "next_node": None}).start()
        
        # Mock all_tools to prevent issues with ToolNode
        self.mock_all_tools = patch('core.tools.all_tools', []).start()

        yield
        patch.stopall() # Clean up all started patches

    @patch('core.graph_builder.StateGraph')
    @patch('core.graph_builder.ToolNode')
    def test_graph_builds_correctly(self, MockToolNode, MockStateGraph):
        # Mock the builder methods
        mock_builder = MagicMock()
        MockStateGraph.return_value = mock_builder
        mock_builder.compile.return_value = MagicMock()

        # Import build_graph here to ensure mocks are active
        from core.graph_builder import build_graph
        build_graph(self.mock_model_with_tools)

        mock_builder.add_node.assert_any_call("load_memories", self.mock_load_memories)
        mock_builder.add_node.assert_any_call("agent", MagicMock()) # The lambda is hard to mock directly, just check it's called
        mock_builder.add_node.assert_any_call("tools", MockToolNode.return_value)
        mock_builder.add_node.assert_any_call("crisis_node", self.mock_crisis_node)
        
        mock_builder.add_edge.assert_any_call(START, "load_memories")
        mock_builder.add_edge.assert_any_call("load_memories", "agent")
        mock_builder.add_conditional_edges.assert_any_call("agent", self.mock_route_tools, ["tools", "crisis_node", END])
        mock_builder.add_edge.assert_any_call("tools", "agent")
        mock_builder.add_edge.assert_any_call("crisis_node", END)
        mock_builder.compile.assert_called_once()

    @patch('core.agent.agent')
    @patch('core.agent.load_memories')
    @patch('core.agent.route_tools')
    def test_message_processing_flow(self, mock_route_tools, mock_load_memories, mock_agent):
        mock_load_memories.return_value = {"recall_memories": ["some_memory"]}
        mock_agent.return_value = {"messages": [AIMessage(content="Agent says hello")], "next_node": None}
        mock_route_tools.return_value = END

        # A simplified graph for testing the flow
        from langgraph.graph import StateGraph, END, START
        from core.state import State
        
        test_builder = StateGraph(State)
        test_builder.add_node("load_memories_node", lambda state, config: mock_load_memories(state, config))
        test_builder.add_node("agent_node", lambda state, config: mock_agent(state, config, MagicMock()))
        test_builder.add_edge(START, "load_memories_node")
        test_builder.add_edge("load_memories_node", "agent_node")
        test_builder.add_conditional_edges("agent_node", lambda state: mock_route_tools(state), {
            "tools": "tools_node", # Mocked as not taken in this test
            "crisis_node": "crisis_node", # Mocked as not taken in this test
            END: END
        })
        test_graph = test_builder.compile()

        user_id = "test_user_flow"
        thread_id = "test_thread_flow"
        initial_state = {"messages": [HumanMessage(content="Hi")]}
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        # Simulate stream processing
        output_chunks = []
        for s in test_graph.stream(initial_state, config):
            output_chunks.append(s)
        
        assert mock_load_memories.called
        assert mock_agent.called
        assert mock_route_tools.called
        
        # Verify final output structure
        final_state = output_chunks[-1]
        assert "agent_node" in final_state
        assert final_state["agent_node"]["messages"][0].content == "Agent says hello"

    @patch('core.agent.agent', return_value={"next_node": "crisis_node"})
    def test_crisis_path_trigger(self, mock_agent):
        # When agent decides to go to crisis_node
        from langgraph.graph import StateGraph, END, START
        from core.state import State
        from core.agent import crisis_node as actual_crisis_node # Import actual crisis_node for side_effect
        
        test_builder = StateGraph(State)
        test_builder.add_node("load_memories_node", lambda state, config: {"recall_memories": []})
        test_builder.add_node("agent_node", lambda state, config: mock_agent(state, config, MagicMock()))
        test_builder.add_node("crisis_node", actual_crisis_node) # Use actual crisis node
        
        test_builder.add_edge(START, "load_memories_node")
        test_builder.add_edge("load_memories_node", "agent_node")
        test_builder.add_conditional_edges("agent_node", lambda state: state["agent_node"]["next_node"], {
            "crisis_node": "crisis_node",
            END: END
        })
        test_builder.add_edge("crisis_node", END)
        
        test_graph = test_builder.compile()

        user_id = "test_user_crisis_trigger"
        thread_id = "test_thread_crisis_trigger"
        initial_state = {"messages": [HumanMessage(content="I'm suicidal")]}
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        # Mock dependencies for actual_crisis_node
        with patch('core.tools.crisis_escalation_tool.invoke') as mock_escalation, \
             patch('core.memory_manager.db') as mock_db_crisis:
            mock_db_crisis.crisis_events.insert_one.return_value = MagicMock()
            mock_escalation.return_value = "Escalation initiated"

            output_chunks = []
            for s in test_graph.stream(initial_state, config):
                output_chunks.append(s)
            
            assert mock_agent.called
            assert mock_escalation.called
            assert mock_db_crisis.crisis_events.insert_one.called
            
            final_state = output_chunks[-1]
            assert "crisis_node" in final_state
            assert "immediate distress" in final_state["crisis_node"]["messages"][0].content

