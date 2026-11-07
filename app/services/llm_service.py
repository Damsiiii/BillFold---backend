"""LLM and Chatbot Service for Personal Finance Q&A.

Supports:
- Gemma 4 E2B / E4B model integration with optional QLoRA adapters
- Lazy singleton model initialization to prevent reloading on each request
- Graceful degradation when model weights are not present or LLM is disabled
- Grounded prompt construction (RAG workflow) preventing financial hallucinations
"""

import logging
import os
from typing import Any, Optional, Tuple
from app.schemas.chatbot import FinancialContext

logger = logging.getLogger(__name__)


class LLMService:
    """Modular LLM service supporting local Gemma models and grounded financial QA."""

    def __init__(self):
        # Configuration from environment variables
        self.enabled: bool = os.getenv("LLM_ENABLED", "false").lower() in ("true", "1", "yes")
        self.model_path: str = os.getenv("LLM_MODEL_PATH", "").strip()
        self.adapter_path: str = os.getenv("LLM_ADAPTER_PATH", "").strip()
        self.device: str = os.getenv("LLM_DEVICE", "cpu")

        # Runtime model handles (lazy loaded)
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._is_fine_tuned: bool = False
        self._initialized: bool = False
        self._load_error: Optional[str] = None

    @property
    def is_fine_tuned(self) -> bool:
        """Returns True ONLY when a confirmed fine-tuned QLoRA adapter is loaded."""
        return self._is_fine_tuned

    def init_model(self) -> bool:
        """Initializes the Gemma model and optional QLoRA adapter lazily.

        Does not download large weights automatically.
        """
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if not self.enabled:
            logger.info("LLM is disabled via LLM_ENABLED=false. Using grounded rule-based provider.")
            return False

        if not self.model_path:
            logger.info("No LLM_MODEL_PATH configured. Running without heavy model.")
            return False

        if not os.path.exists(self.model_path):
            self._load_error = f"Model path does not exist: {self.model_path}"
            logger.warning(self._load_error)
            return False

        try:
            # Conditional imports to ensure backend boots without heavy ML deps
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading Gemma model from: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
            )

            # Check for fine-tuned QLoRA adapter
            if self.adapter_path and os.path.exists(self.adapter_path):
                try:
                    from peft import PeftModel

                    logger.info(f"Loading QLoRA adapter from: {self.adapter_path}")
                    self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
                    self._is_fine_tuned = True
                except Exception as peft_err:
                    logger.warning(f"Failed to load PEFT adapter: {peft_err}. Using base weights.")
                    self._is_fine_tuned = False
            else:
                self._is_fine_tuned = False

            return True

        except Exception as err:
            self._load_error = str(err)
            logger.error(f"Error loading Gemma model: {err}. Falling back to grounded provider.")
            self._model = None
            self._tokenizer = None
            return False

    def construct_rag_prompt(self, user_message: str, context: FinancialContext) -> str:
        """Constructs an augmented prompt embedding retrieved financial context."""
        context_parts = []
        if context.monthly_income is not None:
            context_parts.append(f"- Monthly Income: ${context.monthly_income:,.2f}")
        if context.monthly_expenses is not None:
            context_parts.append(f"- Monthly Total Expenses: ${context.monthly_expenses:,.2f}")
        if context.category_spending:
            cat_str = ", ".join(f"{k}: ${v:,.2f}" for k, v in context.category_spending.items())
            context_parts.append(f"- Category Spending: {cat_str}")
        if context.budget_recommendations:
            context_parts.append(f"- Budget Recommendations: {context.budget_recommendations}")
        if context.savings_recommendations:
            context_parts.append(f"- Savings Recommendations: {context.savings_recommendations}")

        context_block = "\n".join(context_parts) if context_parts else "None available."

        return (
            "System: You are an AI Personal Finance Assistant. "
            "Use ONLY the following verified financial facts. Do NOT invent or fabricate any numbers.\n"
            f"[Verified Financial Data]\n{context_block}\n\n"
            f"User Question: {user_message}\n"
            "Assistant: "
        )

    def generate_response(self, user_message: str, context: FinancialContext) -> Tuple[str, str]:
        """Generates a grounded financial response.

        Returns:
            Tuple[str, str]: (response_text, source)
        """
        lower_msg = user_message.lower()

        # Detect whether the query is seeking the user's specific personal records
        personal_indicators = ["my", " i ", "i'd", "i've", "did i", "have i", "can i afford"]
        has_personal_indicator = any(ind in f" {lower_msg} " for ind in ["my", " i ", "i ", " did i ", " have i "])
        
        financial_terms = [
            "spend", "spent", "spending", "expense", "expenses",
            "income", "earn", "earned", "salary",
            "budget", "save", "saving", "savings",
            "transaction", "transactions", "statement", "balance", "money", "overspend"
        ]
        has_financial_term = any(term in lower_msg for term in financial_terms)

        is_personal_query = (has_personal_indicator and has_financial_term) or any(
            phrase in lower_msg
            for phrase in [
                "how much did i spend",
                "how much have i spent",
                "what did i spend",
                "show my transactions",
                "my spending",
                "my expenses",
                "my income",
                "my budget",
                "my savings",
                "am i overspending",
            ]
        )

        # Guard: User asking for their own data, but no statement/transactions loaded yet
        if is_personal_query and not context.has_data:
            return (
                "Your financial data has not yet been loaded. "
                "Please upload a bank statement first so I can analyze your transactions.",
                "system",
            )

        # If data is available and query is personal, answer with grounded verified figures
        if context.has_data and is_personal_query:
            if any(k in lower_msg for k in ["spend", "spent", "spending", "expense"]):
                if context.monthly_expenses is not None:
                    return (
                        f"Your total spending for this month is ${context.monthly_expenses:,.2f}.",
                        "financial_data",
                    )
            if any(k in lower_msg for k in ["income", "earn", "salary"]):
                if context.monthly_income is not None:
                    return (
                        f"Your recorded income for this month is ${context.monthly_income:,.2f}.",
                        "financial_data",
                    )
            if "budget" in lower_msg and context.budget_recommendations:
                return (
                    f"Based on your transactions, here is your budget status: {context.budget_recommendations}",
                    "financial_data",
                )
            if "saving" in lower_msg and context.savings_recommendations:
                return (
                    f"Here are your personalized savings recommendations: {context.savings_recommendations}",
                    "financial_data",
                )

        # If model is configured and active, invoke it with the RAG prompt
        if self.enabled and self.init_model() and self._model and self._tokenizer:
            try:
                import torch

                prompt = self.construct_rag_prompt(user_message, context)
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_tokens = self._model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.2,
                        do_sample=False,
                    )

                response_text = self._tokenizer.decode(
                    output_tokens[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                source = "financial_data" if context.has_data else "gemma_model"
                return response_text, source
            except Exception as inf_err:
                logger.error(f"Inference error with Gemma model: {inf_err}")
                # Fall through to general fallback

        # General personal finance guidance when no personal records are requested
        # shortcut: minimal rule-based guidance, upgrade with fine-tuned Gemma weights
        if "budget" in lower_msg:
            return (
                "A common budgeting guideline is the 50/30/20 rule: 50% for needs, "
                "30% for wants, and 20% for savings and debt repayment.",
                "general_knowledge",
            )

        if "saving" in lower_msg or "save" in lower_msg:
            return (
                "Building an emergency fund covering 3 to 6 months of essential living "
                "expenses is a foundational personal finance milestone.",
                "general_knowledge",
            )

        return (
            "I am your personal finance assistant. You can ask me questions about your monthly spending, "
            "expenses, budget recommendations, or general finance principles.",
            "general_knowledge",
        )


# Singleton LLM service instance
llm_service = LLMService()
