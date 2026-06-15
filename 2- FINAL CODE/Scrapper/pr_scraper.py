import argparse
import csv
from datetime import datetime, timedelta, timezone
import numpy as np
from github import Github, Auth
import tqdm
from transformers import pipeline
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import re

# NEW: import your fine-tuned BERT inference helper
from use_multitask_bert import predict_sentiment_features


def get_sentiment_batch(texts, sentiment_pipeline):
    if not texts:
        return []
    non_empty = [(i, text) for i, text in enumerate(texts) if text.strip()]
    if not non_empty:
        return [0.0] * len(texts)

    indices, valid_texts = zip(*non_empty)
    results = sentiment_pipeline(
        list(valid_texts),
        batch_size=16,
        truncation=True,
        max_length=512
    )

    sentiments = [0.0] * len(texts)
    for idx, result in zip(indices, results):
        score = result["score"]
        sentiments[idx] = score if result["label"] == "POSITIVE" else -score
    return sentiments


def get_emotion_batch(texts, emotion_pipeline):
    if not texts:
        return []
    non_empty = [(i, text) for i, text in enumerate(texts) if text.strip()]
    if not non_empty:
        return ["neutral"] * len(texts)

    indices, valid_texts = zip(*non_empty)
    results = emotion_pipeline(
        list(valid_texts),
        batch_size=16,
        truncation=True,
        max_length=512,
        top_k=None,
    )

    emotions = ["neutral"] * len(texts)
    for idx, result in zip(indices, results):
        emotions[idx] = max(result, key=lambda x: x["score"])["label"]
    return emotions


def map_emotion_to_category(emotion):
    mapping = {
        "joy": "appreciation",
        "love": "appreciation",
        "anger": "criticism",
        "sadness": "criticism",
        "surprise": "confusion",
        "fear": "urgency",
    }
    return mapping.get(emotion, "confusion")


def estimate_cyclomatic_from_diff(file_additions):
    """Cheap regex-based CC estimate: count decisions in added lines"""
    if not file_additions:
        return 0
    pattern = re.compile(
        r"\b(if|else|for|while|case|switch|try|catch|&&|\|\||\?|=>)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(file_additions)
    return len(set(matches)) + 1


def get_file_extension_lang(filename):
    """Simple lang mapper for distinct_langs"""
    ext = filename.split(".")[-1].lower()
    lang_map = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
        "go": "go",
        "rs": "rust",
        "md": "docs",
        "yml": "yaml",
        "yaml": "yaml",
    }
    return lang_map.get(ext, "other")


def count_todo_fixme_from_patches(added_code_per_file):
    pattern = re.compile(r"\b(TODO|FIXME)\b", re.IGNORECASE)
    count = 0
    for patch in added_code_per_file.values():
        if not patch:
            continue
        count += len(pattern.findall(patch))
    return count


def build_thread_text(pr, comments, review_comments):
    """Build a single concatenated PR thread text for BERT."""
    parts = []
    title = pr.title or ""
    desc = pr.body or ""
    parts.append(f"[TITLE] {title}\n")
    parts.append(f"[DESCRIPTION] {desc}\n")

    # Combine all comments in chronological order
    all_comments = sorted(
        list(comments) + list(review_comments),
        key=lambda c: c.created_at
    )

    for c in all_comments:
        user = c.user.login if c.user else "unknown"
        body = c.body or ""
        # Distinguish review comments vs normal comments in text tags
        tag = "REVIEW_COMMENT" if hasattr(c, "pull_request_url") else "COMMENT"
        parts.append(f"[{tag} by {user}] {body}\n")

    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Scrape PR features from a GitHub repo.")
    parser.add_argument("--repo", required=True, help="Repository in owner/repo format")
    parser.add_argument("--token", required=True, help="GitHub API token")
    parser.add_argument(
        "--start_date",
        default="1900-01-01",
        help="Start date for merged PRs (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max_prs", type=int, default=1000, help="Max number of PRs to scrape"
    )
    parser.add_argument("--output", default="pr_data.csv", help="Output CSV file")
    args = parser.parse_args()

    auth = Auth.Token(args.token)
    g = Github(auth=auth)
    repo = g.get_repo(args.repo)

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    print("Loading sentiment / emotion models...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    emotion_pipeline = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None,
    )
    print("HF models loaded. Fine-tuned BERT will be used via use_multitask_bert.py")

    all_prs = repo.get_pulls(state="closed", sort="updated", direction="desc")
    selected_prs = []
    for pr in all_prs:
        if pr.merged and pr.merged_at >= start_dt:
            selected_prs.append(pr)
            if len(selected_prs) >= args.max_prs:
                break

    print(f"Found {len(selected_prs)} merged PRs after {args.start_date}.")

    print("Fetching repository-level data...")
    contributors = list(repo.get_contributors())
    contrib_counts = [c.contributions for c in contributors]
    top_20_threshold = np.percentile(contrib_counts, 80) if contrib_counts else 0
    contrib_map = {c.login: c.contributions for c in contributors}

    print("Building author experience map...")
    from collections import defaultdict as _dd
    author_pr_counts = _dd(int)
    for pr in selected_prs:
        if pr.user:
            author_pr_counts[pr.user.login] += 1

    data = []
    for idx, pr in enumerate(tqdm.tqdm(selected_prs, desc="Processing PRs")):
        try:
            commits_list = list(pr.get_commits())
            commits_count = len(commits_list)

            lines_added = 0
            lines_deleted = 0
            changed_files = set()
            test_lines_added = 0
            added_code_per_file = defaultdict(str)
            langs_changed = set()

            for commit in commits_list:
                for file in commit.files:
                    lines_added += file.additions
                    lines_deleted += file.deletions
                    changed_files.add(file.filename)
                    if "test" in file.filename.lower():
                        test_lines_added += file.additions
                    if file.additions > 0:
                        added_code_per_file[file.filename] += (file.patch or "").lower()

                    lang = get_file_extension_lang(file.filename)
                    langs_changed.add(lang)

            files_changed = len(changed_files)
            code_churn = lines_added + lines_deleted
            has_tests = any(
                "test" in f.lower() or "_test" in f.lower() for f in changed_files
            )
            test_coverage_change = (
                test_lines_added / lines_added if lines_added > 0 else 0
            )

            total_cc = sum(
                estimate_cyclomatic_from_diff(code)
                for code in added_code_per_file.values()
            )
            cyclomatic_avg = total_cc / files_changed if files_changed > 0 else 0

            lines_changed = lines_added + lines_deleted
            if lines_changed < 100:
                pr_size_category = "S"
            elif lines_changed < 500:
                pr_size_category = "M"
            else:
                pr_size_category = "L"

            comments = list(pr.get_comments())
            review_comments = list(pr.get_review_comments())
            comment_count = len(comments) + len(review_comments)
            review_comment_count = len(review_comments)

            reviews = list(pr.get_reviews())
            num_approvals = sum(1 for review in reviews if review.state == "APPROVED")
            has_approvals = 1 if num_approvals > 0 else 0  # not in final CSV but kept

            # NEW: number of distinct reviewers
            num_reviewers = len({r.user.login for r in reviews if r.user})

            participants = set()
            for comment in comments:
                if comment.user:
                    participants.add(comment.user.login)
            for review_comment in review_comments:
                if review_comment.user:
                    participants.add(review_comment.user.login)
            if pr.user:
                participants.add(pr.user.login)
            participants_count = len(participants)

            author = pr.user
            author_experience = author_pr_counts.get(author.login, 0) - 1 if author else 0
            author_contribs = contrib_map.get(author.login, 0)
            is_core_contributor = 1 if author_contribs >= top_20_threshold else 0
            author_followers = author.followers if author else 0

            merge_delay_days = (
                (pr.merged_at - pr.created_at).days
                if pr.merged_at and pr.created_at
                else 0
            )

            has_ci_passed = 0
            try:
                statuses = list(pr.get_combined_status())
                has_ci_passed = 1 if any(
                    "success" in s.state.lower()
                    for s in statuses
                    if "ci" in s.context.lower() or "test" in s.context.lower()
                ) else 0
            except Exception:
                pass

            commit_msgs = [commit.commit.message for commit in commits_list]
            avg_commit_msg_length = (
                np.mean([len(msg.split()) for msg in commit_msgs])
                if commit_msgs
                else 0
            )

            distinct_langs_changed = len(langs_changed)

            all_comments = sorted(
                comments + review_comments, key=lambda c: c.created_at
            )
            response_times = [
                (all_comments[i].created_at - all_comments[i - 1].created_at).total_seconds()
                / 3600.0
                for i in range(1, len(all_comments))
            ]
            response_time_avg = np.mean(response_times) if response_times else 0

            # NEW: review_wait_time = time from PR created_at to first comment/review
            if all_comments:
                first_event_time = all_comments[0].created_at
                review_wait_time = (
                    (first_event_time - pr.created_at).total_seconds() / 3600.0
                    if pr.created_at
                    else 0
                )
            else:
                review_wait_time = 0

            desc_body = pr.body or ""
            comment_bodies = [c.body or "" for c in comments + review_comments]
            all_texts = [desc_body] + comment_bodies
            desc_idx = 0
            review_start_idx = 1

            # Classic sentiment features from SST2
            all_sentiments = get_sentiment_batch(all_texts, sentiment_pipeline)
            desc_sentiment = all_sentiments[desc_idx]
            review_sentiments = all_sentiments[review_start_idx:]

            review_sentiment_avg = (
                np.mean(review_sentiments) if review_sentiments else 0
            )
            review_sentiment_std = (
                np.std(review_sentiments) if review_sentiments else 0
            )
            most_negative_sentiment = (
                min(review_sentiments) if review_sentiments else 0
            )

            if len(review_sentiments) > 1:
                times = np.arange(len(review_sentiments))
                sentiment_trajectory = np.polyfit(times, review_sentiments, 1)[0]
            else:
                sentiment_trajectory = 0

            # Emotion category using emotion pipeline + mapping
            all_emotions = get_emotion_batch(all_texts, emotion_pipeline)
            mapped_emotions = [map_emotion_to_category(e) for e in all_emotions]
            emotion_categories = (
                Counter(mapped_emotions).most_common(1)[0][0]
                if mapped_emotions
                else "appreciation"
            )

            merge_time = pr.merged_at

            # NEW: TODO/FIXME count from patches
            num_TODO_FIXME = count_todo_fixme_from_patches(added_code_per_file)

            # NEW: code_owner_involvement = any reviewer is core contributor
            core_reviewers = {
                r.user.login
                for r in reviews
                if r.user and contrib_map.get(r.user.login, 0) >= top_20_threshold
            }
            code_owner_involvement = 1 if core_reviewers else 0

            # BERT-based extra sentiment features on full thread text
            thread_text = build_thread_text(pr, comments, review_comments)
            bert_feats = predict_sentiment_features(thread_text)

            # Future bug fixes in 90 days in touched files
            bug_keywords = ["fix", "bug", "error", "defect", "patch"]
            future_commits = list(
                repo.get_commits(
                    since=merge_time, until=merge_time + timedelta(days=90)
                )
            )
            future_bug_fixes = 0
            for commit in future_commits:
                msg_lower = commit.commit.message.lower()
                if any(kw in msg_lower for kw in bug_keywords):
                    if any(f.filename in changed_files for f in commit.files):
                        future_bug_fixes += 1

            # Refactor signal in 180 days
            refactor_keywords = ["refactor", "rewrite", "cleanup", "restructure"]
            long_future_commits = list(
                repo.get_commits(
                    since=merge_time, until=merge_time + timedelta(days=180)
                )
            )
            requires_refactoring = 0
            for commit in long_future_commits:
                msg_lower = commit.commit.message.lower()
                if any(kw in msg_lower for kw in refactor_keywords):
                    for f in commit.files:
                        if f.filename in changed_files and (
                            f.changes > 0.5 * lines_changed if lines_changed > 0 else False
                        ):
                            requires_refactoring = 1
                            break
                if requires_refactoring:
                    break

            # Conflict-ish heuristic (same as before, but a bit safer)
            causes_conflicts = 0
            try:
                # Some libraries expose mergeable_state as string; the previous code assumed attributes
                state = getattr(pr, "mergeable_state", None)
                if isinstance(state, str) and state in ["dirty", "conflicting"]:
                    causes_conflicts = 1
                elif "conflict" in (pr.title or "").lower() or "conflict" in (
                    pr.body or ""
                ).lower():
                    causes_conflicts = 1
            except Exception:
                pass

            # Hotfixes in 7 days in same files
            hotfix_keywords = ["hotfix", "revert", "rollback", "emergency"]
            hotfix_commits = list(
                repo.get_commits(
                    since=merge_time, until=merge_time + timedelta(days=7)
                )
            )
            requires_hotfix = 0
            for commit in hotfix_commits:
                msg_lower = commit.commit.message.lower()
                if any(kw in msg_lower for kw in hotfix_keywords):
                    if any(f.filename in changed_files for f in commit.files):
                        requires_hotfix = 1
                        break

            row = {
                "pr_number": pr.number,
                "merge_date": merge_time.isoformat(),
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "files_changed": files_changed,
                "commits_count": commits_count,
                "comment_count": comment_count,
                "review_comment_count": review_comment_count,
                "participants_count": participants_count,
                "has_tests": int(has_tests),
                "code_churn": code_churn,
                "test_coverage_change": test_coverage_change,
                "cyclomatic_avg": cyclomatic_avg,
                "pr_size_category": pr_size_category,
                "author_experience": author_experience,
                "is_core_contributor": is_core_contributor,
                "author_followers": author_followers,
                "response_time_avg": response_time_avg,
                "description_sentiment": desc_sentiment,
                "review_sentiment_avg": review_sentiment_avg,
                "review_sentiment_std": review_sentiment_std,
                "most_negative_sentiment": most_negative_sentiment,
                "sentiment_trajectory": sentiment_trajectory,
                "emotion_categories": emotion_categories,
                # BERT-based extra features:
                "politeness_score": bert_feats["politeness_score"],
                "uncertainty_score": bert_feats["uncertainty_score"],
                "technical_confidence_score": bert_feats["technical_confidence_score"],
                "reviewer_disagreement_level": bert_feats["reviewer_disagreement_level"],
                "comment_escalation": bert_feats["comment_escalation"],
                "change_justification_clarity": bert_feats[
                    "change_justification_clarity"
                ],
                # New technical features:
                "num_reviewers": num_reviewers,
                "code_owner_involvement": code_owner_involvement,
                "review_wait_time": review_wait_time,
                "num_TODO_FIXME": num_TODO_FIXME,
                "num_approvals": num_approvals,
                "merge_delay_days": merge_delay_days,
                "has_ci_passed": has_ci_passed,
                "avg_commit_msg_length": avg_commit_msg_length,
                "distinct_langs_changed": distinct_langs_changed,
                "future_bug_fixes": future_bug_fixes,
                "requires_refactoring": requires_refactoring,
                "causes_conflicts": causes_conflicts,
                "requires_hotfix": requires_hotfix,
            }

            data.append(row)

        except Exception as e:
            print(f"\nError processing PR #{pr.number}: {e}")
            continue

    if data:
        keys = data[0].keys()
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"\nData saved to {args.output}")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
