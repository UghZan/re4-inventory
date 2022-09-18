using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class InventoryZone : MonoBehaviour
{
    [HideInInspector] public bool isInit;

    [Header("Zone Settings")]
    public Vector2Int gridDim; //lines and columns count
    [SerializeField] int maxItems; //set to -1 to make it fit infinitely
    public ItemSlot[] fittingTypes; //use ALL to make it universal
    
    int gridHeight;

    bool[] occupiedSlots; //slots for fitting inside of a grid
    int slotsTotal; //total amount of slots, lines count x columns count

    List<ItemStack> items;

    [Header("Debug Settings")]
    [SerializeField] bool shouldPrefill;
    [SerializeField] int amount;
    [SerializeField] List<ItemInfo> debugItems;

    public void InitZone()
    {
        if (isInit) return;

        slotsTotal = gridDim.x * gridDim.y;
        occupiedSlots = new bool[slotsTotal];
        items = new List<ItemStack>();
        if(shouldPrefill)
        {
            DebugPrefill();
        }
        isInit = true;
    }

    void DebugPrefill()
    {
        for (int i = 0; i < amount; i++)
        {
            int rnd = Random.Range(0, debugItems.Count);
            AddItem(new ItemStack(debugItems[rnd], Random.Range(1, debugItems[rnd].itemStackSize+1)));
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
    public int GetMaxItems() { return maxItems; }
    public bool IsEmpty => items.Count == 0;
    public bool IsFull => items.Count == maxItems;

    public ItemStack GetFirstItemStack() //useful for 1-fitting inventory zones (such as slots for weapons/armors)
    {
        return GetItemStackAt(0);
    }

    public ItemStack? GetItemStackAt(int index) //useful for 1-fitting inventory zones (such as slots for weapons/armors)
    {
        if(index > items.Count)
        {
            Debug.LogError("Item index outside of capacity");
            return null;
        }
        return items[index];
    }

    public bool AddItem(ItemStack item)
    {
        ItemStack temp = FindNonfullStack(item);
        if (temp != null) //if there's a stack of same item
        {
            int tempAmount = item.GetStackAmount();
            if (temp.UpdateStackAmount(ref tempAmount)) //check if we can fit all of new items in older stack
            {
                item.SetStackAmount(tempAmount);
                return true;
            }
            else
            {
                item.SetStackAmount(tempAmount); //repeat until we get the stack in inventory or run out of place
                AddItem(item);
            }
        }
        else
        {
            if (items.Count == maxItems && maxItems != -1) return false;

            Vector2Int? vacantPos = FindVacantSpot(item.GetRotatedSize().x, item.GetRotatedSize().y, true);
            if (vacantPos == null)
                return false;
            else
            {
                item.SetPositionInZone((Vector2Int)vacantPos);
                item.parentZone = this;
                items.Add(item);
                return true;
            }
        }
        return false;
    }

    public bool AddItemAt(ItemStack itemStack, Vector2Int pos)
    {
        if(CheckIfInvPositionIsFree(pos, itemStack.GetRotatedSize().x, itemStack.GetRotatedSize().y))
        {
            itemStack.SetPositionInZone(pos);
            FillOccupiedSpace(pos, itemStack.GetRotatedSize().x, itemStack.GetRotatedSize().y);
            itemStack.parentZone = this;
            items.Add(itemStack);
            return true;
        }
        return false;
    }

    public bool RemoveItem(ItemStack itemStack)
    {
        if(items.Contains(itemStack))
        {
            items.Remove(itemStack);
            return true;
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
        if (pos.x >= gridDim.x || pos.x + _w - 1 >= gridDim.x) return false;
        if (pos.y * gridDim.x + pos.x + _w - 1 + (_h-1) * gridDim.x > slotsTotal) return false;
        //Debug.Log("not going out of bounds yet");
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
