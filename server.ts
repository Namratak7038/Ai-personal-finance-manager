import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Increase file size limit to handle OCR base64 uploads
app.use(express.json({ limit: "10mb" }));

// Lazy initialisation of Gemini client
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not defined. Please add it to your environment variables or Secrets panel.");
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// 1. Health & Debug Endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    hasApiKey: !!process.env.GEMINI_API_KEY,
    currentTime: new Date().toISOString(),
  });
});

// 2. Receipt Scanner (OCR) Endpoint
app.post("/api/ocr", async (req, res) => {
  try {
    const { imageBase64, mimeType } = req.body;
    if (!imageBase64) {
      return res.status(400).json({ error: "No image content provided." });
    }

    const ai = getGenAI();
    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const imagePart = {
      inlineData: {
        mimeType: mimeType || "image/png",
        data: cleanBase64,
      },
    };

    const promptText = `
      Perform advanced OCR on this receipt or bill image to accurately extract items and financial figures.
      Analyze the text on the receipt and infer details:
      1. amount: the net or gross total payable. If items are listed with their costs, identify the final total paid.
      2. category: classify the primary expense category. Supported categories: 'Food', 'Travel', 'Groceries', 'Shopping', 'Subscriptions', 'Rent', 'Medicine', 'Education', 'Entertainment', 'Insurance', 'Utilities', 'Transfer', or 'Other'.
      3. date: the purchase date in the format YYYY-MM-DD. If year is missing of 2 digits, infer the year as 2026. If no date is found, use today's date "2026-06-16".
      4. description: general merchant name or list of prominent items purchased.
      5. isEssential: boolean indicating whether it covers primary needs (food, rent, groceries, medicine, education) as opposed to optional purchases (shopping, entertainment, subscriptions).
      
      Return a neatly formatted JSON compliance structure.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [imagePart, { text: promptText }],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            amount: { type: Type.NUMBER, description: "Extracted total amount (number only)" },
            category: { type: Type.STRING, description: "Primary category based on matching list" },
            date: { type: Type.STRING, description: "Date of transaction formatted as YYYY-MM-DD" },
            description: { type: Type.STRING, description: "Name of the partner, store, or key items" },
            isEssential: { type: Type.BOOLEAN, description: "True if essentials, false if optional/luxury spend" },
          },
          required: ["amount", "category", "date", "description", "isEssential"],
        },
      },
    });

    const text = response.text || "{}";
    const data = JSON.parse(text.trim());
    return res.json({ success: true, data });
  } catch (error: any) {
    console.error("OCR API error:", error);
    return res.status(500).json({
      error: "Failed to scan receipt image.",
      details: error.message || String(error),
    });
  }
});

// 3. AI Financial Advisor Endpoint
app.post("/api/advisor", async (req, res) => {
  try {
    const { expenses, budget, goals, currentLanguage = "en" } = req.body;
    const ai = getGenAI();

    const dataContext = JSON.stringify({ expenses, budget, goals });
    const promptText = `
      You are an expert AI Financial Advisor. Analyze the user's spending habits, monthly budget (currently limit is ₹${budget || 0}), goals, and categories.
      
      Generate a thorough, practical personal financial review and save-money action plan.
      Please deliver the response in ${currentLanguage === "hi" ? "Hindi (हिंदी)" : currentLanguage === "mr" ? "Marathi (मराठी)" : "English"}.
      Even if the output is in Hindi or Marathi, ensure it matches the requested JSON structures exactly.
      
      Identify spending patterns:
      - Distinguish deep essentials (groceries, medicine, rent) from luxuries/optionals (shopping, entertainment).
      - Alert on excessive spending categories of highest values.
      - Draft budget revisions and smart tips.
      - Calculate a realistic "Financial Health Score" from 0 to 100 based on standard ratios (e.g. keeping savings high, shopping/entertainment low).
      - Make sure you identify recurring expenses and point them out if they aren't marked as recurring.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [
        { text: `Data Payload: ${dataContext}` },
        { text: promptText }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            savingsPotential: { type: Type.STRING, description: "Monthly potential savings amount/text in selected language" },
            essentialSpendTotal: { type: Type.NUMBER, description: "Calculated sum of essential expenses" },
            optionalSpendTotal: { type: Type.NUMBER, description: "Calculated sum of optional luxury expenses" },
            unnecessaryExpenses: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "List of unneeded item expenses or areas to curb"
            },
            alerts: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "High alert statements regarding budget excession or fast spending"
            },
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING, description: "Brief title in selected language" },
                  description: { type: Type.STRING, description: "Action step details in selected language" },
                  category: { type: Type.STRING },
                  priority: { type: Type.STRING, description: "High, Medium, or Low urgency" },
                },
                required: ["title", "description", "category", "priority"]
              }
            },
            savingsTips: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Generic list of smart frugality tips"
            },
            financialHealthScore: { type: Type.NUMBER, description: "Score out of 100 representing user health status" }
          },
          required: [
            "savingsPotential", "essentialSpendTotal", "optionalSpendTotal",
            "unnecessaryExpenses", "alerts", "recommendations", "savingsTips",
            "financialHealthScore"
          ],
        },
      }
    });

    const text = response.text || "{}";
    const data = JSON.parse(text.trim());
    return res.json({ success: true, data });
  } catch (error: any) {
    console.error("Advisor API error:", error);
    return res.status(500).json({
      error: "Advisor analysis failed.",
      details: error.message || String(error),
    });
  }
});

// 4. Future Expense Prediction Endpoint
app.post("/api/predict", async (req, res) => {
  try {
    const { expenses, budget = 10000, currentLanguage = "en" } = req.body;
    const ai = getGenAI();

    const dataContext = JSON.stringify(expenses);
    const promptText = `
      You are an Expense forecasting engine. Based on the user's historical spending logs list, predict:
      1. predictedNextWeekSpend: expected overall spending over the upcoming 7 days.
      2. predictedNextMonthSpend: expected overall spending over the next 30 days.
      3. likelihoodToExceedBudget: % probability (integer 0-100) that they exceed their monthly budget limit (which is currently ₹${budget}).
      4. categoryForecasts: forecast predicted values for each category (Food, Travel, Groceries, Shopping, Subscriptions, Rent, Medicine, Education, Entertainment, Insurance, Utilities, Transfer, Other). Compare current period spending with predicted period.
      5. insightsMarkdown: a robust descriptive text summary of these future predictions, potential risks, and recommendations. Provide this markdown content in ${currentLanguage === "hi" ? "Hindi (हिंदी)" : currentLanguage === "mr" ? "Marathi (मराठी)" : "English"}.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [
        { text: `Expense Logs: ${dataContext}` },
        { text: promptText }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            predictedNextWeekSpend: { type: Type.NUMBER },
            predictedNextMonthSpend: { type: Type.NUMBER },
            likelihoodToExceedBudget: { type: Type.INTEGER },
            categoryForecasts: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  category: { type: Type.STRING },
                  currentPeriodSpend: { type: Type.NUMBER },
                  predictedPeriodSpend: { type: Type.NUMBER },
                  riskLevel: { type: Type.STRING, description: "Low, Medium, or High" }
                },
                required: ["category", "currentPeriodSpend", "predictedPeriodSpend", "riskLevel"]
              }
            },
            insightsMarkdown: { type: Type.STRING, description: "Descriptive forecasting analysis" }
          },
          required: ["predictedNextWeekSpend", "predictedNextMonthSpend", "likelihoodToExceedBudget", "categoryForecasts", "insightsMarkdown"]
        }
      }
    });

    const text = response.text || "{}";
    const data = JSON.parse(text.trim());
    return res.json({ success: true, data });
  } catch (error: any) {
    console.error("Prediction API error:", error);
    return res.status(500).json({
      error: "Prediction analysis failed.",
      details: error.message || String(error),
    });
  }
});

// 5. Goal Savings advice planner Endpoint
app.post("/api/goal-advice", async (req, res) => {
  try {
    const { goal, monthlySurplus = 0, currentLanguage = "en" } = req.body;
    const ai = getGenAI();

    const promptText = `
      Analyze this savings goal:
      Goal name: "${goal.name}"
      Target Amount: ₹${goal.targetAmount}
      Current Saved: ₹${goal.currentAmount || 0}
      Deadline: ${goal.deadline}
      Current monthly budget surplus (Income - Expenses): ₹${monthlySurplus}

      Calculate:
      1. monthlySavingsRequired: target amount divided by remaining months.
      2. dailySavingsRequired: divided by remaining days.
      3. feasibility: "Easy", "Moderate", "Hard", or "Realistically Unachievable" based on their current monthly surplus.
      4. actionPlan: Step-by-step action plan to reach this goal.
      5. aiSuggestions: Custom hacks (frugal substitutions, side streams) to collect the sum faster.
      
      Deliver actionPlan and aiSuggestions in ${currentLanguage === "hi" ? "Hindi (हिंदी)" : currentLanguage === "mr" ? "Marathi (मराठी)" : "English"}.
      Use standard YYYY-MM-DD current year parameters (which is 2026).
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [{ text: promptText }],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            monthlySavingsRequired: { type: Type.NUMBER },
            dailySavingsRequired: { type: Type.NUMBER },
            feasibility: { type: Type.STRING },
            actionPlan: { type: Type.ARRAY, items: { type: Type.STRING } },
            aiSuggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
            daysForecastToAchievement: { type: Type.NUMBER, description: "Days estimated to finish goal under current saving rate" }
          },
          required: ["monthlySavingsRequired", "dailySavingsRequired", "feasibility", "actionPlan", "aiSuggestions", "daysForecastToAchievement"]
        }
      }
    });

    const text = response.text || "{}";
    const data = JSON.parse(text.trim());
    return res.json({ success: true, data });
  } catch (error: any) {
    console.error("Goal Advice API error:", error);
    return res.status(500).json({
      error: "Failed to generate savings roadmap.",
      details: error.message || String(error),
    });
  }
});

// 6. Gemini Chatbot Endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, userContext, currentLanguage = "en" } = req.body;
    const ai = getGenAI();

    const recentMessage = messages[messages.length - 1];
    const previousHistory = messages.slice(0, -1).map((m: any) => ({
      role: m.sender === "user" ? "user" : "model",
      parts: [{ text: m.text }]
    }));

    const systemInstruction = `
      You are Finny, a friendly but highly skilled digital AI Personal Finance Assistant.
      The user is querying you inside their personal expense tracking app.
      
      Below is the user's live financial data context:
      - Monthly Budget Limit: ₹${userContext.budget || 10000}
      - Total Current Spending: ₹${userContext.totalSpend || 0}
      - Active Savings Target Goals: ${JSON.stringify(userContext.goals || [])}
      - Recent Expense Records List: ${JSON.stringify(userContext.recentExpenses || [])}
      - Recurring schedules: ${JSON.stringify(userContext.recurring || [])}
      - Current Year: 2026.
      
      Respond directly to user queries using clear parameters. Mention exact calculations, budget health status, and suggest specific actions.
      Keep responses brief, actionable, and encouraging.
      IMPORTANT: You MUST write your complete response in ${currentLanguage === "hi" ? "Hindi (हिंदी)" : currentLanguage === "mr" ? "Marathi (मराठी)" : "English"}.
      Feel free to use Indian Rupee (₹) symbol.
    `;

    const chatSession = ai.chats.create({
      model: "gemini-3.5-flash",
      config: {
        systemInstruction,
        temperature: 0.8
      },
      history: previousHistory
    });

    const response = await chatSession.sendMessage({ message: recentMessage.text });

    // Suggest 3 follow-up prompts based on the current explanation
    const promptSuggestorText = `
      Based on the chatbot response: "${response.text}", output exactly 3 brief relevant follow-up questions a user might click.
      Format your response as a simple JSON string list.
      Language must be ${currentLanguage === "hi" ? "Hindi (हिंदी)" : currentLanguage === "mr" ? "Marathi (मराठी)" : "English"}.
    `;

    let suggestions: string[] = [];
    try {
      const suggestorResponse = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: [{ text: promptSuggestorText }],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          }
        }
      });
      suggestions = JSON.parse(suggestorResponse.text || "[]");
    } catch {
      suggestions = [
        "How can I cut more expenses?",
        "Are my essentials too high?",
        "Show my highest spending category"
      ];
    }

    return res.json({
      success: true,
      text: response.text,
      suggestions: suggestions.slice(0, 3)
    });
  } catch (error: any) {
    console.error("Chat API error:", error);
    return res.status(500).json({
      error: "Chat assistant experienced issues.",
      details: error.message || String(error),
    });
  }
});

// 7. Simulating Cloud Sync Endpoint
app.post("/api/sync", (req, res) => {
  const { expenses, expensesCount, goalsCount, recurringCount, timestamp } = req.body;
  console.log(`Cloud Sync initiated at ${timestamp || new Date().toISOString()}:`, {
    expensesSynced: expensesCount,
    goalsSynced: goalsCount,
    recurringSynced: recurringCount,
  });
  return res.json({
    success: true,
    message: "Global cloud database structures synced successfully.",
    syncId: "sync_" + Math.random().toString(36).substr(2, 9),
    serverTime: new Date().toISOString(),
  });
});

// Configure Vite or Static Assets serving based on Environment
async function startViteMiddleware() {
  if (process.env.NODE_ENV !== "production") {
    // Development Mode
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production Mode
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[AI Personal Expense Tracker Server] boot completed on http://localhost:${PORT}`);
  });
}

startViteMiddleware();
