"""Product-text rules for identifying unresolved annotations."""
from __future__ import annotations

from dataclasses import dataclass

from phanet.common.io import digest_object


def normalized_product(product):
    if not isinstance(product, str):
        raise ValueError("A present product string is required; missing records are errors")
    return " ".join(product.casefold().split())


@dataclass(frozen=True)
class BaselineDecision:
    unresolved: bool
    reason: str
    rule_id: str


class ProductTextBaseline:
    """Normalize product text, apply exact-description overrides, then match text rules."""

    def __init__(self, policy):
        if (not isinstance(policy, dict) or type(policy.get("schema")) is not int or policy["schema"] != 1
                or not isinstance(policy.get("rule_id"), str) or not policy["rule_id"].strip()
                or policy.get("scope") != "development_pilot"
                or policy.get("parent_rule") != "pilot-product-text-v1"):
            raise ValueError("Invalid annotation policy schema, scope or parent rule")
        for field in ("generic", "substrings", "prefixes"):
            if not isinstance(policy.get(field), list) or any(not isinstance(v, str) for v in policy[field]):
                raise ValueError(f"Baseline {field} must be a list of strings")
        if not isinstance(policy.get("reviewed_descriptions"), list):
            raise ValueError("Baseline reviewed_descriptions must be a list")
        self.rule_id = policy["rule_id"]
        self.generic = frozenset(normalized_product(v) for v in policy["generic"])
        self.substrings = tuple(normalized_product(v) for v in policy["substrings"])
        self.prefixes = tuple(normalized_product(v) for v in policy["prefixes"])
        if not all(self.substrings + self.prefixes):
            raise ValueError("Empty substring/prefix would classify every description")
        self.overrides = {}
        for entry in policy["reviewed_descriptions"]:
            if (not isinstance(entry, dict) or not isinstance(entry.get("product"), str) or
                    type(entry.get("unresolved")) is not bool or not isinstance(entry.get("reason"), str) or
                    not entry["reason"].strip()):
                raise ValueError("Reviewed decisions require product text, a boolean and rationale")
            key = normalized_product(entry["product"])
            decision = (entry["unresolved"], entry["reason"])
            if key in self.overrides and self.overrides[key] != decision:
                raise ValueError("Conflicting normalized description decisions")
            self.overrides[key] = decision
        self.signature = digest_object(policy)

    def assess(self, product):
        key = normalized_product(product)
        if key in self.overrides:
            unresolved, reason = self.overrides[key]
            return BaselineDecision(unresolved, "reviewed:" + reason, self.rule_id)
        if (key in self.generic or any(v in key for v in self.substrings)
                or key.startswith(self.prefixes)):
            return BaselineDecision(True, "parent_unresolved_text", self.rule_id)
        return BaselineDecision(False, "parent_descriptive_text_unreviewed", self.rule_id)

    def assess_locus(self, products):
        decisions = [self.assess(product) for product in products]
        if not decisions:
            raise ValueError("Missing locus products must not become unresolved silently")
        return BaselineDecision(all(d.unresolved for d in decisions),
                                "all_products_unresolved" if all(d.unresolved for d in decisions)
                                else "at_least_one_descriptive_product", self.rule_id)
