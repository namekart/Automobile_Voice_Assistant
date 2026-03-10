from __future__ import annotations

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, JobContext, room_io
from livekit.plugins import deepgram, openai, silero, sarvam, noise_cancellation
# from livekit.plugins import sarvam

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful automobile customer support voice assistant.
            You answer questions about vehicle service, help users book service appointments,
            and can escalate to a human when needed. Be concise and clear. Avoid emojis and complex formatting.""",
        )


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt = sarvam.STT(
            model="saaras:v3",           # Use the SOTA V3 model
            language="unknown",          # Set to unknown for Automatic Language Detection
            mode="translate",            # CRITICAL: Converts any Indic speech to English text
            high_vad_sensitivity=True,   # Helps in noisy automobile workshops
            flush_signal=True            # Faster turn-taking
        ),
        llm=openai.LLM.with_openrouter(model="openai/gpt-4o-mini"),
        tts=sarvam.TTS(
            model="bulbul:v3",
            target_language_code="en-IN",
            speaker="shubh",
            pace=1.1,
            speech_sample_rate=24000,
        ),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance with automobile support."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
