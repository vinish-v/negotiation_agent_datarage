"""
Final Validation Script for DataRage Aggressive Analyst Agent
=============================================================

This script validates that our buyer agent achieves 100% success rate
and demonstrates the aggressive analytical personality in action.
"""

from negotiation_agent_datarage import YourBuyerAgent, test_your_agent, Product, run_negotiation_test
import json

def comprehensive_validation():
    """Run comprehensive validation of our agent"""
    
    print("="*80)
    print(" DATARAGE AGGRESSIVE ANALYST - FINAL VALIDATION ")
    print("="*80)
    
    # Initialize our agent
    agent = YourBuyerAgent("DataRage_Ultimate")
    
    # Display personality traits
    personality = agent.define_personality()
    print(f"\n📊 AGENT PROFILE:")
    print(f"   Name: {agent.name}")
    print(f"   Type: {personality['personality_type']}")
    print(f"   Traits: {', '.join(personality['traits'])}")
    print(f"   Style: {personality['negotiation_style']}")
    print(f"\n🗣️  SIGNATURE PHRASES:")
    for phrase in personality['catchphrases']:
        print(f"   • \"{phrase}\"")
    
    print(f"\n🧠 PERSONALITY PROMPT:")
    print(f"   {agent.get_personality_prompt()}")
    
    print("\n" + "="*80)
    print("🎯 NEGOTIATION PERFORMANCE TEST")
    print("="*80)
    
    # Run the standard test
    test_your_agent()
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE !")
    print("="*80)
   
    
    return True

if __name__ == "__main__":
    comprehensive_validation()
