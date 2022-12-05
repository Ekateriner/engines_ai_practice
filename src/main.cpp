#include "raylib.h"
#include <functional>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <algorithm>
#include <chrono>
#include <thread>
#include "math.h"
#include "dungeonGen.h"
#include "dungeonUtils.h"
#include <iostream>

template <typename T>
static size_t coord_to_idx(T x, T y, size_t w)
{
  return size_t(y) * w + size_t(x);
}

static void draw_nav_grid(const char *input, size_t width, size_t height)
{
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
    {
      char symb = input[coord_to_idx(x, y, width)];
      Color color = GetColor(symb == ' ' ? 0xeeeeeeff : symb == 'o' ? 0x7777ffff
                                                                    : 0x222222ff);
      DrawPixel(int(x), int(y), color);
    }
}

static void draw_path(std::vector<Position> path)
{
  for (const Position &p : path)
    DrawPixel(p.x, p.y, GetColor(0x44000088));
}

static std::vector<Position> reconstruct_path(std::vector<Position> prev, Position to, size_t width)
{
  Position curPos = to;
  std::vector<Position> res = {curPos};
  while (prev[coord_to_idx(curPos.x, curPos.y, width)] != Position{-1, -1})
  {
    curPos = prev[coord_to_idx(curPos.x, curPos.y, width)];
    res.insert(res.begin(), curPos);
  }
  return res;
}

static std::vector<Position> find_path_a_star(const char *input, size_t width, size_t height, Position from, Position to)
{
  if (from.x < 0 || from.y < 0 || from.x >= int(width) || from.y >= int(height))
    return std::vector<Position>();
  size_t inpSize = width * height;

  std::vector<float> g(inpSize, std::numeric_limits<float>::max());
  std::vector<float> f(inpSize, std::numeric_limits<float>::max());
  std::vector<Position> prev(inpSize, {-1, -1});

  auto getG = [&](Position p) -> float
  { return g[coord_to_idx(p.x, p.y, width)]; };
  auto getF = [&](Position p) -> float
  { return f[coord_to_idx(p.x, p.y, width)]; };

  auto heuristic = [](Position lhs, Position rhs) -> float
  {
    return sqrtf(square(float(lhs.x - rhs.x)) + square(float(lhs.y - rhs.y)));
  };

  g[coord_to_idx(from.x, from.y, width)] = 0;
  f[coord_to_idx(from.x, from.y, width)] = heuristic(from, to);

  using queue_item = std::pair<float, Position>;
  auto cmp = [](auto r, auto l)
  { return r.first > l.first; };
  std::priority_queue<queue_item, std::vector<queue_item>, decltype(cmp)> openList(cmp);
  std::vector<bool> inQueue(inpSize, false);
  openList.push({0.0f, from});
  inQueue[coord_to_idx(from.x, from.y, width)] = true;

  std::vector<bool> closedList(inpSize, false);

  while (!openList.empty())
  {
    Position curPos = openList.top().second;
    size_t idx = coord_to_idx(curPos.x, curPos.y, width);
    openList.pop();
    inQueue[idx] = false;

    if (curPos == to)
      return reconstruct_path(prev, to, width);
    if (closedList[idx])
      continue;
    DrawPixel(curPos.x, curPos.y, Color{uint8_t(g[idx]), uint8_t(g[idx]), 0, 100});
    closedList[idx] = true;
    auto checkNeighbour = [&](Position p)
    {
      // out of bounds
      if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height))
        return;
      size_t p_idx = coord_to_idx(p.x, p.y, width);
      // not empty
      if (input[p_idx] == '#')
        return;
      float weight = input[p_idx] == 'o' ? 10.f : 1.f;
      float gScore = getG(curPos) + 1.f * weight; // we're exactly 1 unit away
      if (gScore < getG(p))
      {
        prev[p_idx] = curPos;
        g[p_idx] = gScore;
        f[p_idx] = gScore + heuristic(p, to);
      }
      if (!inQueue[p_idx])
      {
        openList.push({getF(p), p});
        inQueue[p_idx] = true;
      }
    };
    checkNeighbour({curPos.x + 1, curPos.y + 0});
    checkNeighbour({curPos.x - 1, curPos.y + 0});
    checkNeighbour({curPos.x + 0, curPos.y + 1});
    checkNeighbour({curPos.x + 0, curPos.y - 1});
  }
  // empty path
  return std::vector<Position>();
}

static std::vector<Position> find_path_ada_star(const char *input, size_t width, size_t height, Position from, Position to, bool new_path)
{
  static auto time = std::chrono::steady_clock::now();
  static std::vector<Position> path = std::vector<Position>();

  static std::vector<std::pair<Position, float>> pixels;
  auto draw_pixels = [&]() -> void {
    for (auto [curPos, g] : pixels) {
      DrawPixel(curPos.x, curPos.y, Color{uint8_t(g), uint8_t(g), 0, 100});
    }
  };
  if (std::chrono::steady_clock::now() - time < std::chrono::seconds(1) && !new_path) {
    draw_pixels();
    return path;
  }
  pixels.clear();

  if (from.x < 0 || from.y < 0 || from.x >= int(width) || from.y >= int(height)) {
    return std::vector<Position>();
  }
  size_t inpSize = width * height;

  static std::vector<float> g(inpSize, std::numeric_limits<float>::max());
  static std::vector<float> f(inpSize, std::numeric_limits<float>::max());
  static std::vector<Position> prev(inpSize, {-1, -1});

  auto getG = [&](Position p) -> float
  { return g[coord_to_idx(p.x, p.y, width)]; };
  auto getF = [&](Position p) -> float
  { return f[coord_to_idx(p.x, p.y, width)]; };

  auto heuristic = [](Position lhs, Position rhs) -> float
  {
    return sqrtf(square(float(lhs.x - rhs.x)) + square(float(lhs.y - rhs.y)));
  };

  static float eps = 2.0;

  auto calc_f = [&](Position p) {
    return getG(p) + eps * heuristic(p, to);
  };

  auto queue_cmp = [&](auto l, auto r) { return calc_f(l) > calc_f(r); };
  static std::vector<Position> openList;
  static std::vector<bool> inQueue(inpSize, false);

  static std::vector<bool> closedList(inpSize, false);
  static std::vector<Position> incons;

  auto push = [&](Position p) -> void
  {
    openList.push_back(p);
    std::push_heap(openList.begin(), openList.end(), queue_cmp);
    inQueue[coord_to_idx(p.x, p.y, width)] = true;
  };

  auto top_pop = [&] -> Position
  {
    auto p = openList.front();
    std::pop_heap(openList.begin(), openList.end(), queue_cmp);
    openList.pop_back();
    inQueue[coord_to_idx(p.x, p.y, width)] = false;
    closedList[coord_to_idx(p.x, p.y, width)] = true;
    return p;
  };

  auto improve_path = [&]() -> std::vector<Position> {
    while (calc_f(to) > calc_f(openList.front())) {
      Position curPos = top_pop();
      size_t idx = coord_to_idx(curPos.x, curPos.y, width);

      pixels.push_back({curPos, g[idx]});
      auto checkNeighbour = [&](Position p)
      {
        // out of bounds
        if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height))
          return;
        size_t p_idx = coord_to_idx(p.x, p.y, width);
        // not empty
        if (input[p_idx] == '#')
          return;
        float weight = input[p_idx] == 'o' ? 10.f : 1.f;
        float gScore = getG(curPos) + 1.f * weight; // we're exactly 1 unit away
        if (gScore < getG(p))
        {
          prev[p_idx] = curPos;
          g[p_idx] = gScore;
          f[p_idx] = gScore + eps * heuristic(p, to);
          if (!closedList[p_idx])
          {
            push(p);
          }
          else {
            incons.push_back(p);
          }
        }
      };
      checkNeighbour({curPos.x + 1, curPos.y + 0});
      checkNeighbour({curPos.x - 1, curPos.y + 0});
      checkNeighbour({curPos.x + 0, curPos.y + 1});
      checkNeighbour({curPos.x + 0, curPos.y - 1});
    }
    return reconstruct_path(prev, to, width);
  };

  auto pure_f = [&](Position val) {
    return getG(val) + heuristic(val, to);
  };
  auto calc_eps = [&]() {
    if (incons.empty() && openList.empty())
      return eps;
    if (incons.empty())
      return std::min(eps, getG(to) / std::ranges::min(openList | std::views::transform(pure_f)));
    if (openList.empty())
      return std::min(eps, getG(to) / std::ranges::min(incons   | std::views::transform(pure_f)));
    return std::min(eps, 
                    getG(to) / std::min(
                                std::ranges::min(openList | std::views::transform(pure_f)),
                                std::ranges::min(incons   | std::views::transform(pure_f)) ));
  };

  // alg
  if (new_path) {
    g = std::vector<float>(inpSize, std::numeric_limits<float>::max());
    f = std::vector<float>(inpSize, std::numeric_limits<float>::max());
    prev = std::vector<Position>(inpSize, {-1, -1});
    eps = 2.0;

    openList.clear();
    inQueue = std::vector<bool>(inpSize, false);
    closedList = std::vector<bool> (inpSize, false);
    incons.clear();

    g[coord_to_idx(from.x, from.y, width)] = 0;
    f[coord_to_idx(from.x, from.y, width)] = heuristic(from, to);
    push(from);
    inQueue[coord_to_idx(from.x, from.y, width)] = true;
    path = std::vector<Position>();
    pixels.clear();
  }
  float eps_current = eps;

  if (eps_current > 1)
  {
    std::ranges::copy(incons, std::back_inserter(openList));
    incons.clear();
    std::make_heap(openList.begin(), openList.end(), queue_cmp);
    closedList.clear();

    path = improve_path();
    eps_current = calc_eps();
    eps /= 1.1;
  }

  time = std::chrono::steady_clock::now();
  return path;
}

static std::vector<Position> find_path_sma_star(const char *input, size_t width, size_t height, Position from, Position to, int max_memmory = 100)
{
  if (from.x < 0 || from.y < 0 || from.x >= int(width) || from.y >= int(height))
    return std::vector<Position>();
  size_t inpSize = width * height;

  std::vector<float> g(inpSize, std::numeric_limits<float>::max());
  std::vector<float> f(inpSize, std::numeric_limits<float>::max());
  std::vector<Position> prev(inpSize, {-1, -1});

  auto getG = [&](Position p) -> float
  { return g[coord_to_idx(p.x, p.y, width)]; };
  auto getF = [&](Position p) -> float
  { return f[coord_to_idx(p.x, p.y, width)]; };

  auto heuristic = [](Position lhs, Position rhs) -> float
  {
    return sqrtf(square(float(lhs.x - rhs.x)) + square(float(lhs.y - rhs.y)));
  };

  g[coord_to_idx(from.x, from.y, width)] = 0;
  f[coord_to_idx(from.x, from.y, width)] = heuristic(from, to);

  using queue_item = std::tuple<float, Position, int, int>;
  auto queue_cmp = [](auto l, auto r)
  { return std::pair(std::get<0>(l), -std::get<2>(l)) > std::pair(std::get<0>(r), -std::get<2>(r)); };
  std::vector<queue_item> openList;
  std::vector<bool> inQueue(inpSize, false);

  auto push = [&](queue_item value) -> void
  {
    openList.push_back(value);
    Position p = std::get<1>(value);
    std::push_heap(openList.begin(), openList.end(), queue_cmp);
    inQueue[coord_to_idx(p.x, p.y, width)] = true;
  };

  auto top = [&] -> std::tuple<Position, int, int>
  {
    auto top_element = openList.front();
    std::get<3>(openList.front()) += 1;
    return {std::get<1>(top_element), std::get<2>(top_element), std::get<3>(top_element)};
  };

  auto pop = [&](auto it) -> void
  {
    if (it == openList.end())
      return;
    Position p = std::get<1>(*it);
    std::get<0>(*it) = std::numeric_limits<float>::min();
    std::push_heap(openList.begin(), it + 1, queue_cmp);
    std::pop_heap(openList.begin(), openList.end(), queue_cmp);
    openList.pop_back();
    inQueue[coord_to_idx(p.x, p.y, width)] = false;
  };

  push({0.0f, from, 0, 0});
  inQueue[coord_to_idx(from.x, from.y, width)] = true;

  std::vector<bool> closedList(inpSize, false);

  std::function<void(Position)> backup = [&](Position p) -> void
  {
    int p_idx = coord_to_idx(p.x, p.y, width);
    Position parent = prev[p_idx];
    if (closedList[p_idx] && parent != Position{-1, -1})
    {
      float f_val = f[p_idx];
      std::array<Position, 4> successors = {Position{p.x + 1, p.y + 0},
                                            Position{p.x - 1, p.y + 0},
                                            Position{p.x + 0, p.y + 1},
                                            Position{p.x + 0, p.y - 1}};
      f[p_idx] = std::ranges::min(successors | std::views::transform([&](auto pos)
                                                                     { return getF(pos); }));
      if (f_val != f[p_idx])
        backup(parent);
    }
  };
  int used = 1;
  auto relax = [&]() -> void
  {
    used += 1;
    if (used > max_memmory)
    {
      auto it = std::min_element(openList.begin(), openList.end(),
                                 [](auto l, auto r)
                                 {
                                   return std::make_pair(std::get<2>(l), -std::get<0>(l)) < std::make_pair(std::get<2>(r), -std::get<0>(r));
                                 });
      Position deleted = std::get<1>(*it);
      int depth = std::get<2>(*it);
      // remove
      pop(it);

      // ?
      // f[coord_to_idx(deleted.x, deleted.y, width)] = std::numeric_limits<float>::max();
      // auto parent = prev[coord_to_idx(deleted.x, deleted.y, width)];
      // if (!inQueue[coord_to_idx(parent.x, parent.y, width)])
      //   push({getF(parent), parent, depth - 1});
      used -= 1;
    }
  };

  auto getNeighbour = [](Position p, int ind) -> Position {
    std::array<Position, 4> successors = {Position{p.x + 1, p.y + 0},
                                          Position{p.x - 1, p.y + 0},
                                          Position{p.x + 0, p.y + 1},
                                          Position{p.x + 0, p.y - 1}};
    return successors[ind];
  };

  auto allChecked = [&](Position pos) -> bool {
    std::array<Position, 4> successors = {Position{pos.x + 1, pos.y + 0},
                                          Position{pos.x - 1, pos.y + 0},
                                          Position{pos.x + 0, pos.y + 1},
                                          Position{pos.x + 0, pos.y - 1}};
    return std::all_of(successors.begin(), successors.end(), [&](auto p) 
            {return inQueue[coord_to_idx(p.x, p.y, width)] || closedList[coord_to_idx(p.x, p.y, width)];} );
  };

  while (!openList.empty())
  {
    auto [curPos, curDepth, curIdx] = top();
    //std::cout << curIdx << " " << getG(curPos) << "\n";
    size_t idx = coord_to_idx(curPos.x, curPos.y, width);

    if (curPos == to)
      return reconstruct_path(prev, to, width);
    DrawPixel(curPos.x, curPos.y, Color{uint8_t(g[idx]), uint8_t(g[idx]), 0, 100});
    auto checkNeighbour = [&](Position p)
    {
      // out of bounds
      if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height))
        return;
      size_t p_idx = coord_to_idx(p.x, p.y, width);
      // not empty
      if (input[p_idx] == '#') {
        closedList[p_idx] = true;
        return;
      }
      
      if (curDepth + 1 == max_memmory) {
        f[p_idx] = std::numeric_limits<float>::max();
        return;
      }
      
      float weight = input[p_idx] == 'o' ? 10.f : 1.f;
      float gScore = getG(curPos) + 1.f * weight; // we're exactly 1 unit away 
      
      if (gScore < getG(p))
      {
        prev[p_idx] = curPos;
        g[p_idx] = gScore;
        f[p_idx] = std::max(getF(curPos), gScore + heuristic(p, to));
      }

      if (!(inQueue[p_idx] || closedList[p_idx]))
      {
        relax();
        push({getF(p), p, curDepth + 1, 0});
      }
    };
    if (curIdx == 3) {
      backup(curPos);
    }
    if (allChecked(curPos) || curIdx == 3) {
      pop(std::find_if(openList.begin(), openList.end(), [curPos](queue_item val) {return std::get<1>(val) == curPos;}));
      closedList[idx] = true;
    }
    checkNeighbour(getNeighbour(curPos, curIdx));
  }
  // empty path
  return std::vector<Position>();
}

// void draw_nav_data(const char *input, size_t width, size_t height, Position from, Position to)
// {
//   draw_nav_grid(input, width, height);
//   std::vector<Position> path = find_path_sma_star(input, width, height, from, to);
//   draw_path(path);
// }

void draw_nav_data(const char *input, size_t width, size_t height, Position from, Position to, bool new_path)
{
  draw_nav_grid(input, width, height);
  std::vector<Position> path = find_path_ada_star(input, width, height, from, to, new_path);
  draw_path(path);
}


int main(int /*argc*/, const char ** /*argv*/)
{
  int width = 1920;
  int height = 1080;
  InitWindow(width, height, "w3 AI MIPT");

  const int scrWidth = GetMonitorWidth(0);
  const int scrHeight = GetMonitorHeight(0);
  if (scrWidth < width || scrHeight < height)
  {
    width = std::min(scrWidth, width);
    height = std::min(scrHeight - 150, height);
    SetWindowSize(width, height);
  }

  constexpr size_t dungWidth = 100;
  constexpr size_t dungHeight = 100;
  char *navGrid = new char[dungWidth * dungHeight];
  gen_drunk_dungeon(navGrid, dungWidth, dungHeight, 24, 100);
  spill_drunk_water(navGrid, dungWidth, dungHeight, 8, 10);

  Position from = dungeon::find_walkable_tile(navGrid, dungWidth, dungHeight);
  Position to = dungeon::find_walkable_tile(navGrid, dungWidth, dungHeight);
  bool action = true;

  Camera2D camera = {{0, 0}, {0, 0}, 0.f, 1.f};
  // camera.offset = Vector2{ width * 0.5f, height * 0.5f };
  camera.zoom = float(height) / float(dungHeight);

  SetTargetFPS(60); // Set our game to run at 60 frames-per-second
  while (!WindowShouldClose())
  {
    // pick pos
    Vector2 mousePosition = GetScreenToWorld2D(GetMousePosition(), camera);
    Position p{int(mousePosition.x), int(mousePosition.y)};
    if (IsMouseButtonPressed(2) || IsKeyPressed(KEY_Q))
    {
      size_t idx = coord_to_idx(p.x, p.y, dungWidth);
      if (idx < dungWidth * dungHeight)
        navGrid[idx] = navGrid[idx] == ' ' ? '#' : navGrid[idx] == '#' ? 'o' : ' ';
      action = true;
    }
    else if (IsMouseButtonPressed(0))
    {
      Position &target = from;
      target = p;
      action = true;
    }
    else if (IsMouseButtonPressed(1))
    {
      Position &target = to;
      target = p;
      action = true;
    }
    else if (IsKeyPressed(KEY_SPACE))
    {
      gen_drunk_dungeon(navGrid, dungWidth, dungHeight, 24, 100);
      spill_drunk_water(navGrid, dungWidth, dungHeight, 8, 10);
      from = dungeon::find_walkable_tile(navGrid, dungWidth, dungHeight);
      to = dungeon::find_walkable_tile(navGrid, dungWidth, dungHeight);
      action = true;
    }
    BeginDrawing();
      ClearBackground(BLACK);
      BeginMode2D(camera);
        draw_nav_data(navGrid, dungWidth, dungHeight, from, to, action);
      EndMode2D();
    EndDrawing();
    action = false;
  }
  delete[] navGrid;
  CloseWindow();
  return 0;
}
