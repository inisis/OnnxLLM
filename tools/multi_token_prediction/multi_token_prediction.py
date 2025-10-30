import time
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MiMoMultiTokenInference:
    """
    Inference engine for MiMo model with built-in multi-token prediction.

    This class implements speculative decoding using MiMo's multi-token prediction (MTP)
    layers to generate multiple tokens per forward pass, then verifies them against the
    main model for accuracy.

    Attributes:
        device: Device to run the model on ('cuda' or 'cpu')
        tokenizer: HuggingFace tokenizer for the model
        model: The MiMo causal language model
        num_mtp_layers: Number of multi-token prediction layers in the model
    """
    def __init__(
        self,
        model_path: str = "/data/llm/MiMo-7B-Base",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="auto"
        )

        self.model.to(self.device)
        self.model.eval()

        # Get number of MTP layers
        self.num_mtp_layers = self.model.config.num_nextn_predict_layers
        print(f"Loaded MiMo model with {self.num_mtp_layers} MTP layers")

        # Metrics tracking
        self.reset_metrics()

        # KV cache for efficient generation
        self.past_key_values: Optional[torch.Tensor] = None
        self.past_key_values_main: Optional[torch.Tensor] = None
        self.past_key_values_draft: Optional[torch.Tensor] = None

    def reset_metrics(self) -> None:
        """Reset generation metrics."""
        self.total_mtp_predictions: int = 0
        self.accepted_mtp_predictions: int = 0
        self.generation_steps: int = 0
        self.tokens_generated: int = 0
        self.generation_time: float = 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics including accept rate."""
        accept_rate = (
            self.accepted_mtp_predictions / self.total_mtp_predictions
            if self.total_mtp_predictions > 0
            else 0.0
        )
        return {
            "accept_rate": accept_rate,
            "total_predictions": self.total_mtp_predictions,
            "accepted_predictions": self.accepted_mtp_predictions,
            "generation_steps": self.generation_steps,
            "tokens_generated": self.tokens_generated,
            "generation_time": self.generation_time,
            "tokens_per_second": (
                self.tokens_generated / self.generation_time
                if self.generation_time > 0
                else 0.0
            ),
            "avg_accepted_per_step": (
                self.accepted_mtp_predictions / self.generation_steps
                if self.generation_steps > 0
                else 0.0
            )
        }

    @torch.no_grad()
    def generate_with_mtp(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        use_cache: bool = False,
    ) -> str:
        """
        Generate text using MiMo's multi-token prediction layers with speculative decoding.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            use_cache: Whether to use KV-cache for efficient generation

        Returns:
            Generated text including the prompt
        """
        # Reset metrics for this generation
        self.reset_metrics()
        self.past_key_values = None

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Time the generation
        start_time = time.time()

        # Use MiMo's native MTP capabilities for generation
        generated = self._generate_mtp(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )

        # Record generation time
        self.generation_time = time.time() - start_time

        # Decode and return
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _generate_mtp(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Internal generation using MTP layers for multi-token prediction.

        The main model only runs during:
        1. Prefill stage - processes the initial prompt
        2. Verification stage - validates draft tokens from MTP layers

        Args:
            input_ids: Tokenized input sequence
            max_new_tokens: Maximum number of new tokens to generate
            use_cache: Whether to use KV-cache for efficient generation

        Returns:
            Complete generated token sequence including input
        """
        tokens_generated = 0

        # Prefill: Run main model once on the initial prompt
        outputs = self.model.model(
            input_ids,
            past_key_values=self.past_key_values,
            use_cache=use_cache
        )
        hidden_states = outputs.last_hidden_state
        main_logits = self.model.lm_head(hidden_states[:, -1:, :])
        main_token = main_logits[:, -1, :].argmax(dim=-1)
        current_abs_position = input_ids.size(1)

        # Prefill draft model
        if self.num_mtp_layers > 0:
            self._proposal_stage(input_ids, hidden_states, 0, use_cache=use_cache)

        if use_cache:
            generated_tokens = torch.cat([input_ids.clone(), main_token.unsqueeze(1)], dim=1)
            input_ids = main_token.unsqueeze(1)
            hidden_states = hidden_states[:, -1:, :]
            self.past_key_values_main = outputs.past_key_values
            self.past_key_values_draft = self.past_key_values_draft
        else:
            input_ids = torch.cat([input_ids, main_token.unsqueeze(1)], dim=1)
            hidden_states = torch.cat([hidden_states, hidden_states[:, -1:, :]], dim=1)
            generated_tokens = input_ids.clone()

        while tokens_generated < max_new_tokens:
            self.generation_steps += 1

            if self.num_mtp_layers > 0:
                output_tokens, hidden_states = self._speculative_decoding(
                    input_ids, hidden_states, current_abs_position, use_cache
                )
                num_new_tokens = output_tokens.size(1)
                current_abs_position += num_new_tokens
                tokens_generated += num_new_tokens
                self.tokens_generated += num_new_tokens
                generated_tokens = torch.cat([generated_tokens, output_tokens], dim=1)
                if use_cache:
                    input_ids = output_tokens
                    kv_cache_len = self.past_key_values_main.get_seq_length()
                    if num_new_tokens == 1:
                        self.past_key_values_main.crop(kv_cache_len - 1) # invalidate bonus cache
                else:
                    input_ids = generated_tokens
                    current_abs_position = input_ids.size(1)
            else:
                outputs = self.model.model(input_ids, past_key_values=self.past_key_values, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state
                self.past_key_values = outputs.past_key_values
                main_logits = self.model.lm_head(hidden_states[:, -1:, :])
                input_ids = main_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_tokens = torch.cat([generated_tokens, input_ids], dim=1)
                tokens_generated += 1
                self.tokens_generated += 1

        self.past_key_values = None
        self.past_key_values_main = None
        self.past_key_values_draft = None
        return generated_tokens

    def _speculative_decoding(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        current_abs_position: int,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform speculative decoding with three stages: proposal, score, and verify.

        Args:
            input_ids: Current token sequence
            hidden_states: Current hidden states from main model
            current_abs_position: Current absolute position in the sequence
            use_cache: Whether to use KV-cache

        Returns:
            Tuple of (output_tokens, updated_hidden_states)
        """
        # Stage 1: Proposal - Build full draft sequence
        draft_sequence, hidden_states_draft = self._proposal_stage(input_ids, hidden_states, current_abs_position, use_cache)
        self.total_mtp_predictions += 1

        # Stage 2: Score - Run main model once to get logits and hidden states
        # get last token from input_ids because others are cached
        if use_cache:
            score_tokens = torch.cat([input_ids[:, -1:], draft_sequence.unsqueeze(0)], dim=1)
        else:
            score_tokens = torch.cat([input_ids, draft_sequence.unsqueeze(0)], dim=1)
        verify_logits, hidden_states_main = self._score_stage(score_tokens, use_cache)

        # Stage 3: Verify - Compare predictions and accept/reject
        output_tokens, hidden_states = self._verify_stage(
            draft_sequence, verify_logits, hidden_states_main
        )

        if use_cache:
            hidden_states_draft = hidden_states
        else:
            hidden_states_draft = torch.cat([hidden_states_draft, hidden_states], dim=1)

        return output_tokens, hidden_states_draft

    def _proposal_stage(
        self,
        current_sequence: torch.Tensor,
        hidden_states: torch.Tensor,
        start_pos: int,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Generate draft tokens using MTP layers.

        Uses the multi-token prediction layers to propose candidate next tokens
        based on the current sequence and hidden states.

        Args:
            current_sequence: Current token sequence
            hidden_states: Hidden states from the main model
            start_pos: Starting position for positional encoding
            use_cache: Whether to use KV-cache

        Returns:
            Tuple of (draft_sequence, mtp_hidden_states)
        """
        batch_size = hidden_states.size(0)
        input_embeds = self.model.model.embed_tokens(current_sequence)

        if use_cache:
            end_pos = input_embeds.size(1) + start_pos
        else:
            start_pos = 0
            end_pos = input_embeds.size(1)

        position_ids = torch.arange(
            start_pos, end_pos, dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.model.model.rotary_emb(input_embeds, position_ids)

        outputs = self.model.model.mtp_layers[0](
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            use_cache=use_cache,
            past_key_values=self.past_key_values_draft,
        )
        mtp_hidden_states = outputs.last_hidden_state
        self.past_key_values_draft = outputs.past_key_values

        # Predict next token from the last position
        logits = self.model.lm_head(mtp_hidden_states[:, -1, :])
        draft_sequence = logits.argmax(dim=-1)

        return draft_sequence, mtp_hidden_states

    def _score_stage(
        self,
        draft_sequence: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Score draft tokens using the main model.

        Runs the main model on the draft sequence to get verification predictions.

        Args:
            draft_sequence: Draft tokens to verify
            use_cache: Whether to use KV-cache

        Returns:
            Tuple of (verify_sequence, hidden_states)
        """
        # Run main model ONCE on the full draft sequence
        outputs = self.model.model(
            draft_sequence, use_cache=use_cache, past_key_values=self.past_key_values_main
        )
        hidden_states = outputs.last_hidden_state
        self.past_key_values_main = outputs.past_key_values

        # Get logits for all positions
        verify_logits = self.model.lm_head(hidden_states)
        verify_sequence = verify_logits.argmax(dim=-1)

        return verify_sequence, hidden_states

    def _verify_stage(
        self,
        draft_sequence: torch.Tensor,
        verify_sequence: torch.Tensor,
        hidden_states_main: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: Verify draft tokens against main model predictions.

        Compares the MTP draft predictions with the main model's predictions
        and either accepts the draft tokens or uses the main model's predictions.

        Args:
            draft_sequence: Draft tokens from MTP layers
            verify_sequence: Verification tokens from main model
            hidden_states_main: Hidden states from main model

        Returns:
            Tuple of (accepted_tokens, corresponding_hidden_states)
        """

        verify_sequence = verify_sequence[:, -2:]
        non_spec_tokens = verify_sequence[:, :1]
        hidden_states_main = hidden_states_main[:, -2:, :]

        if torch.all(draft_sequence == non_spec_tokens):
            # MTP prediction is correct - accept it
            output_tokens = verify_sequence
            self.accepted_mtp_predictions += 1
            hidden_states = hidden_states_main
        else:
            # Mismatch: use main model's prediction instead
            output_tokens = non_spec_tokens
            hidden_states = hidden_states_main[:, :1, :]

        # print(f"Current accept rate: {self.accepted_mtp_predictions}/{self.total_mtp_predictions} = {self.accepted_mtp_predictions/self.total_mtp_predictions:.1%}")

        return output_tokens, hidden_states


def _run_benchmark(
    inference: MiMoMultiTokenInference,
    prompt: str,
    num_tokens: int,
    use_mtp: bool,
    use_cache: bool,
) -> Tuple[str, Dict[str, float]]:
    """
    Run a single benchmark iteration.

    Args:
        inference: The MiMo inference engine
        prompt: Text prompt to generate from
        num_tokens: Number of tokens to generate
        use_mtp: Whether to use multi-token prediction
        use_cache: Whether to use KV-cache

    Returns:
        Tuple of (generated_text, metrics_dict)
    """
    if not use_mtp:
        original_mtp = inference.num_mtp_layers
        inference.num_mtp_layers = 0

    text = inference.generate_with_mtp(
        prompt=prompt, max_new_tokens=num_tokens, use_cache=use_cache
    )

    if not use_mtp:
        inference.num_mtp_layers = original_mtp

    return text, inference.get_metrics()


def compare_generation_speeds() -> None:
    """
    Compare generation speeds between standard and MTP modes.

    Benchmarks three modes:
    1. Standard generation without MTP
    2. MTP with verification (no cache)
    3. MTP with verification (with cache)

    Prints detailed timing statistics and acceptance rates.
    """
    inference = MiMoMultiTokenInference(model_path="/data/llm/MiMo-7B-Base")

    prompt = "Today is"
    num_tokens = 20
    num_runs = 1

    print(f"Prompt: {prompt}")
    print(f"Generating {num_tokens} tokens, {num_runs} runs each\n")
    print("=" * 80)

    # Benchmark 1: Standard generation (no MTP)
    print("\n[1] Standard Generation (no MTP, with cache)")
    print("  Warm-up run...")
    _run_benchmark(inference, prompt, num_tokens, use_mtp=False, use_cache=False)

    standard_times = []
    for i in range(num_runs):
        text, metrics = _run_benchmark(
            inference, prompt, num_tokens, use_mtp=False, use_cache=False
        )
        print(text)
        standard_times.append(metrics["generation_time"])
        print(
            f"  Run {i+1}: {metrics['generation_time']:.3f}s "
            f"({metrics['tokens_per_second']:.2f} tok/s)"
        )

    avg_standard_time = sum(standard_times) / len(standard_times)
    avg_standard_tps = num_tokens / avg_standard_time

    # Benchmark 2: MTP with verification (no cache)
    print("\n[2] MTP with Verification")
    print("  Warm-up run...")
    _run_benchmark(inference, prompt, num_tokens, use_mtp=True, use_cache=False)

    mtp_times = []
    for i in range(num_runs):
        text, metrics = _run_benchmark(
            inference, prompt, num_tokens, use_mtp=True, use_cache=False
        )
        print(text)
        mtp_times.append(metrics["generation_time"])
        print(
            f"  Run {i+1}: {metrics['generation_time']:.3f}s "
            f"({metrics['tokens_per_second']:.2f} tok/s, "
            f"accept: {metrics['accept_rate']:.1%})"
        )

    avg_mtp_time = sum(mtp_times) / len(mtp_times)
    avg_mtp_tps = num_tokens / avg_mtp_time

    # Benchmark 3: MTP with verification (with cache)
    print("\n[3] MTP with Verification with cache")
    print("  Warm-up run...")
    _run_benchmark(inference, prompt, num_tokens, use_mtp=True, use_cache=True)

    mtp_cache_times = []
    for i in range(num_runs):
        text, metrics = _run_benchmark(
            inference, prompt, num_tokens, use_mtp=True, use_cache=True
        )
        print(text)
        mtp_cache_times.append(metrics["generation_time"])
        print(
            f"  Run {i+1}: {metrics['generation_time']:.3f}s "
            f"({metrics['tokens_per_second']:.2f} tok/s, "
            f"accept: {metrics['accept_rate']:.1%})"
        )

    avg_mtp_cache_time = sum(mtp_cache_times) / len(mtp_cache_times)
    avg_mtp_cache_tps = num_tokens / avg_mtp_cache_time

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Mode':<35} {'Avg Time':<12} {'Tok/s':<12} {'Speedup'}")
    print("-" * 80)
    print(
        f"{'Standard (no MTP)':<35} {avg_standard_time:.3f}s      "
        f"{avg_standard_tps:6.2f}       1.00x"
    )
    print(
        f"{'MTP with Verification':<35} {avg_mtp_time:.3f}s      "
        f"{avg_mtp_tps:6.2f}       {avg_standard_time/avg_mtp_time:.2f}x"
    )
    print(
        f"{'MTP with Verification (cache)':<35} {avg_mtp_cache_time:.3f}s      "
        f"{avg_mtp_cache_tps:6.2f}       {avg_standard_time/avg_mtp_cache_time:.2f}x"
    )
    print("\n" + "=" * 80)

    # Get final metrics from last run
    final_metrics = inference.get_metrics()
    if final_metrics["total_predictions"] > 0:
        print(f"\nMTP Accept Rate (last run): {final_metrics['accept_rate']:.1%}")
        print(
            f"Avg MTP tokens accepted per step: "
            f"{final_metrics['avg_accepted_per_step']:.2f}"
        )
        print(f"Total MTP layers: {inference.num_mtp_layers}")


if __name__ == "__main__":
    compare_generation_speeds()
