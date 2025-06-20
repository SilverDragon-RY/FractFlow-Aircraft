#!/usr/bin/env python3
"""
ui.py
Description: UI implementation for FractFlow using NiceGUI
"""

import asyncio
from datetime import datetime
from typing import List, Tuple, Dict, Any
from uuid import uuid4

from nicegui import ui

from FractFlow.agent import Agent


class FractFlowUI:
    """UI implementation for FractFlow using NiceGUI"""
    
    def __init__(self, agent: Agent):
        """Initialize the UI with an agent instance"""
        self.agent = agent
        self.messages: List[Tuple[str, str, str, str, List[Dict[str, Any]]]] = []  # Added history field
        self.user_id = str(uuid4())
        self.bot_id = "agent"
        self._is_initialized = False
        self._loading_indicator = None
        
        # Setup UI page
        @ui.page('/')
        async def main():
            await self._setup_ui()

    async def initialize(self):
        """Initialize the agent"""
        if not self._is_initialized:
            await self.agent.initialize()
            self._is_initialized = True

    async def _setup_ui(self):
        """Setup the UI components"""
        # Setup header
        with ui.header().classes('bg-primary text-white'):
            ui.label('FractFlow Chat').classes('text-h5 q-px-md q-py-sm')
        
        # Setup chat area
        with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
            self._setup_chat_messages()
        
        # Setup input area
        with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
            self._setup_input_area()
            
        # Setup loading indicator
        self._loading_indicator = ui.spinner(size='lg').classes('absolute bottom-4 right-4')
        self._loading_indicator.visible = False

    @ui.refreshable
    def _chat_messages(self):
        """Display chat messages"""
        if self.messages:
            for user_id, avatar, text, stamp, history in self.messages:
                # For user messages (no history to display)
                if user_id == self.user_id:
                    ui.chat_message(
                        text=text,
                        stamp=stamp,
                        avatar=avatar,
                        sent=True
                    )
                # For bot messages with history
                else:
                    with ui.chat_message(
                        stamp=stamp,
                        avatar=avatar,
                        sent=False
                    ):
                        ui.markdown(text)
                        
                        # Add details expansion if we have history
                        if history:
                            with ui.expansion('查看详细处理过程', icon='visibility').classes('w-full mt-2'):
                                self._render_history_details(history)
        else:
            ui.label('No messages yet').classes('mx-auto my-36')
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

    def _render_history_details(self, history: List[Dict[str, Any]]):
        """Render the detailed history of agent's processing"""
        # Group history by iterations for better readability
        iteration = 0
        current_iteration_expansion = None # Keep track of the current expansion panel
        
        for entry in history:
            role = entry.get('role')
            content = entry.get('content', '')
            tool_calls = entry.get('tool_calls', [])
            
            # Start of a new user query marks a new conversation
            if role == 'user':
                # Display user query directly in the expansion (not ideal maybe, but simple)
                # Or potentially handle this outside the iteration loop if structure changes
                with ui.card().classes('w-full my-2 bg-blue-50'):
                    ui.markdown(f"**用户查询:** {content}")
                current_iteration_expansion = None # Reset iteration on new query
                iteration = 0
                continue
                
            # If this is an assistant message with tool calls, start a new iteration expansion
            if role == 'assistant' and tool_calls:
                iteration += 1
                # Create the expansion panel for this iteration
                current_iteration_expansion = ui.expansion(f"迭代 {iteration}", icon='loop').classes('w-full my-1')

            # Ensure we are inside an iteration expansion before adding cards
            if current_iteration_expansion:
                with current_iteration_expansion:
                    # Prepare content based on role
                    if role == 'assistant':
                        # Assistant's thinking/response
                        with ui.card().classes('w-full my-1 bg-green-50'):
                            ui.markdown(f"**助手思考:**\\n```\\n{content}\\n```")
                            
                            # If there are tool calls, show them
                            if tool_calls:
                                ui.markdown("**工具调用:**")
                                for tc in tool_calls:
                                    func = tc.get('function', {})
                                    tc_id = tc.get('id', 'unknown')
                                    with ui.card().classes('my-1 bg-purple-50'):
                                        ui.markdown(f"- **工具:** `{func.get('name')}`")
                                        ui.markdown(f"- **ID:** `{tc_id}`")
                                        ui.markdown(f"- **参数:** \\n```json\\n{func.get('arguments')}\\n```")
                                        
                    elif role == 'tool':
                        # Tool execution result
                        name = entry.get('name', 'unknown')
                        tool_call_id = entry.get('tool_call_id', 'unknown')
                        
                        with ui.card().classes('w-full my-1 bg-yellow-50'):
                            ui.markdown(f"**工具结果 (`{name}`):**")
                            ui.markdown(f"**ID:** `{tool_call_id}`")
                            # Show full results
                            result_str = str(content)
                            ui.markdown(f"```\\n{result_str}\\n```")
            else:
                 # Handle cases where we might get tool results before the first assistant message with tool_calls
                 # Or handle the final assistant message without tool calls
                 if role == 'assistant' and not tool_calls:
                     # Display final thought/response outside iterations? Or in a final 'Result' section?
                      with ui.card().classes('w-full my-1 bg-green-100'):
                         ui.markdown(f"**最终思考/回复 (无工具调用):**\\n```\\n{content}\\n```")
                 elif role == 'tool':
                      # This case might be less common if history is ordered correctly
                      with ui.card().classes('w-full my-1 bg-yellow-100'):
                          name = entry.get('name', 'unknown')
                          tool_call_id = entry.get('tool_call_id', 'unknown')
                          ui.markdown(f"**工具结果 (`{name}`) (迭代外?):**")
                          ui.markdown(f"**ID:** `{tool_call_id}`")
                          result_str = str(content)
                          ui.markdown(f"```\\n{result_str}\\n```")

    def _setup_chat_messages(self):
        """Setup the chat messages display"""
        self._chat_messages()

    def _setup_input_area(self):
        """Setup the input area"""
        with ui.row().classes('w-full no-wrap items-center'):
            with ui.avatar():
                ui.image(f'https://robohash.org/{self.user_id}?bgset=bg2')
            text_input = ui.input(placeholder='Ask the FractFlow Agent...') \
                .props('rounded outlined input-class=mx-3').classes('flex-grow')
            # Pass the input element itself to the handler
            text_input.on('keydown.enter', lambda: asyncio.create_task(self._handle_message(text_input)))
            ui.button('Send', on_click=lambda: asyncio.create_task(self._handle_message(text_input))) \
                .props('icon=send')
        
        ui.markdown('Built with [FractFlow](https://github.com/yourusername/FractFlow) and [NiceGUI](https://nicegui.io)') \
            .classes('text-xs self-end mr-8 m-[-1em] text-primary')

    async def _handle_message(self, text_input: ui.input):
        """Handle user message, now taking the input element"""
        message_text = text_input.value # Get the current value
        if not message_text.strip():
            return
            
        # Clear the input field *before* potentially long-running agent processing
        text_input.value = ''
        
        # Add user message to the display
        self._add_user_message(message_text)
        
        # Show loading indicator
        self._loading_indicator.visible = True
        
        try:
            # Process with agent using the captured text
            result = await self.agent.process_query(message_text)
            # Get history from agent after processing
            history = self.agent.get_history()
            self._add_bot_message(result, history)
        except Exception as e:
            self._add_error_message(str(e))
        finally:
            # Hide loading indicator
            self._loading_indicator.visible = False

    def _add_user_message(self, text: str):
        """Add a user message"""
        self.messages.append((
            self.user_id,
            f'https://robohash.org/{self.user_id}?bgset=bg2',
            text,
            datetime.now().strftime('%X'),
            []  # Empty history for user messages
        ))
        self._chat_messages.refresh()

    def _add_bot_message(self, text: str, history: List[Dict[str, Any]] = None):
        """Add a bot message with optional history"""
        if history is None:
            history = []
            
        self.messages.append((
            self.bot_id,
            f'https://robohash.org/{self.bot_id}?bgset=bg1',
            text,
            datetime.now().strftime('%X'),
            history
        ))
        self._chat_messages.refresh()

    def _add_error_message(self, error: str):
        """Add an error message"""
        self.messages.append((
            self.bot_id,
            f'https://robohash.org/{self.bot_id}?bgset=bg1',
            f"Error: {error}",
            datetime.now().strftime('%X'),
            []  # No history for error messages
        ))
        self._chat_messages.refresh()

    async def shutdown(self):
        """Shutdown the UI and agent"""
        if self._is_initialized:
            await self.agent.shutdown()
            self._is_initialized = False

    @staticmethod
    def run():
        """Run the UI server"""
        ui.run(title="FractFlow Chat") 