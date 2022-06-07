using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class InventoryManager : MonoBehaviour
{
    [Header("Grid Settings")]
    [SerializeField] Vector2Int gridDim; //lines and columns count
    int gridHeight;

    bool[] occupiedSlots; //slots for fitting inside of a grid
    int slotsTotal; //total amount of slots, lines count x columns count

    List<ItemStack> items;

    [Header("Debug Settings")]
    //testing
    public bool debugAdd;
    public ItemInfo[] testItems;
    public int howMuch;
    void Start()
    {
        slotsTotal = gridDim.x * gridDim.y;
        occupiedSlots = new bool[slotsTotal];
        items = new List<ItemStack>();
        if(debugAdd)
        {
            TestItems();
        }
    }

    void TestItems()
    {
        for (int i = 0; i < howMuch; i++)
        {
            AddItem(this, new ItemStack(testItems[UnityEngine.Random.Range(0, testItems.Length)], 1));
        }
    }

    ItemStack FindNonfullStack(ItemStack subject)
    {
        foreach (ItemStack stack in items)
        {
            if (stack.item == subject.item && !stack.IsFull())
            {
                return stack;
            }
        }
        return null;
    }

    public int GetGridHeight() { return gridHeight; }

    public static bool AddItem(InventoryManager inv, ItemStack item)
    {
        ItemStack temp = inv.FindNonfullStack(item);
        if (temp != null) //if there's a stack of same item
        {
            int tempAmount = item.amount;
            if (temp.UpdateStack(ref tempAmount)) //check if we can fit all of new items in older stack
            {
                item.SetAmount(tempAmount);
                return true;
            }
            else
            {
                item.SetAmount(tempAmount); //repeat until we get the stack in inventory or run out of place
                AddItem(inv, item);
            }
        }
        else
        {
            Vector2Int? vacantPos = inv.FindVacantSpot(item.item.itemSize.x, item.item.itemSize.y, true);
            if (vacantPos == null)
                return false;
            else
            {
                item.invPos = (Vector2Int)vacantPos;
                inv.items.Add(item);
                return true;
            }
        }
        return false;
    }

    public List<ItemStack> GetItemList()
    {
        return items;
    }

    //if setPlace is true, will place item in the occupied grid, else will just return a place for an item
    public Vector2Int? FindVacantSpot(int _w, int _h, bool setPlace = false)
    {
        bool check = true;
        int lastSlot = 0;
        for (int i = 0; i < slotsTotal; i++)
        {
            if (!occupiedSlots[i]) //found an empty spot
            {
                for (int j = 0; j < _w && check; j++)
                {
                    for (int k = 0; k < _h && check; k++)
                    {
                        if (Mathf.FloorToInt((i + j) / gridDim.x) != Mathf.FloorToInt(i / gridDim.x)) check = false;
                        if (i + j + k * gridDim.x >= slotsTotal)
                        {
                            check = false;
                            return null;
                        }
                        if (occupiedSlots[i + j + k * gridDim.x])
                            check = false;
                    }
                }
                if (check)
                {
                    Vector2Int p = new Vector2Int(i % gridDim.x, i / gridDim.x);
                    if (setPlace)
                    {
                        for (int j = 0; j < _w; j++)
                        {
                            for (int k = 0; k < _h; k++)
                            {
                                occupiedSlots[i + j + k * gridDim.x] = true;
                            }
                        }
                        if (i + (_w - 1) + (_h - 1) * gridDim.x > lastSlot)
                            lastSlot = Mathf.CeilToInt((i + (_w - 1) + (_h - 1) * gridDim.x) / gridDim.x);

                        if (lastSlot > gridHeight)
                        {
                            gridHeight = lastSlot;
                        }
                    }
                    return p;
                }
                check = true;
            }
        }
        return null;
    }

    public bool CheckIfInvPositionIsFree(Vector2Int pos, int _w, int _h)
    {
        if (pos.y * gridDim.x + pos.x + _w - 1 + _h - 1 > slotsTotal) return false;
        for (int j = 0; j < _w; j++)
        {
            for (int k = 0; k < _h; k++)
            {
                if (occupiedSlots[pos.x + pos.y * gridDim.x + j + k * gridDim.x]) return false;
            }
        }
        return true;
    }

    public void ClearOccupiedSpace(Vector2Int pos, int _w, int _h)
    {
        for (int j = 0; j < _w; j++)
        {
            for (int k = 0; k < _h; k++)
            {
                occupiedSlots[pos.x + pos.y * gridDim.x + j + k * gridDim.x] = false;
            }
        }
    }

    public void FillOccupiedSpace(Vector2Int pos, int _w, int _h)
    {
        for (int j = 0; j < _w; j++)
        {
            for (int k = 0; k < _h; k++)
            {
                occupiedSlots[pos.x + pos.y * gridDim.x + j + k * gridDim.x] = true;
            }
        }
    }
}
